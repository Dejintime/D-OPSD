from contextlib import nullcontext, contextmanager
import inspect

import torch
from peft import LoraConfig, get_peft_model
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import model_inputs, response_logits


class ModelBundle:
    """One frozen backbone; two named PEFT adapters, never active together.

    Both parameter groups keep requires_grad=True: PEFT's set_adapter normally
    disables inactive parameters, which would invalidate cross-role meta gradients.
    Fixed-teacher forwards support non-reentrant checkpoints. Meta forwards keep
    recomputation disabled and can offload saved activations to host RAM.
    """

    def __init__(self, base, rank=8, alpha=16, targets=None):
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                            target_modules=targets or ["q_proj", "v_proj"],
                            task_type="CAUSAL_LM", bias="none")
        self.model = get_peft_model(base, config, adapter_name="student")
        self.model.add_adapter("teacher", config)
        self._restore_grad_flags()
        self.model.eval()  # eval controls dropout, not autograd
        self.model.config.use_cache = False
        self.supports_selected_logits = "logits_to_keep" in inspect.signature(base.forward).parameters
        self.gradient_checkpointing = False
        self.attention_overrides = {}
        self.offload_teacher_activations = False
        self.offload_inner_activations = False
        self.meta_gradient = 'unroll'

    def student_jvp(self, view, direction):
        """Directional logit derivative without recording a reverse/Hessian graph."""
        self.model.set_adapter('student')
        self._restore_grad_flags()
        params = self.parameters('student')
        kwargs = dict(model_inputs(view),use_cache=False)
        if self.supports_selected_logits:
            kwargs['logits_to_keep']=torch.arange(view['start'],view['start']+view['length'],device=self.device)
        previous=self.model.config._attn_implementation
        self.model.config._attn_implementation='eager'  # Flash has no forward AD.
        def function(values):
            result=functional_call(self.model,values,(),kwargs,strict=False)
            return result.logits if self.supports_selected_logits else response_logits(result,view)
        try:
            with torch.no_grad():
                return torch.func.jvp(function,(params,),(direction,))
        finally:
            self.model.config._attn_implementation=previous

    @contextmanager
    def attention_context(self, role, params):
        key = 'quiz' if role == 'student' and params is not None else role
        backend = self.attention_overrides.get(key)
        previous = self.model.config._attn_implementation
        if backend is not None:
            self.model.config._attn_implementation = backend
        try:
            yield
        finally:
            self.model.config._attn_implementation = previous

    @contextmanager
    def activation_offload(self):
        # Parameters already remain resident: copying them on every save wastes
        # host RAM without freeing device memory. Use pageable CPU activation
        # buffers and preserve broadcast views instead of allocating dense copies.
        resident = {p.untyped_storage().data_ptr() for p in self.model.parameters()}
        def pack(tensor):
            if tensor.device.type != 'cuda' or tensor.untyped_storage().data_ptr() in resident:
                return tensor.detach()
            value = tensor.detach()
            for dim,(size,stride) in enumerate(zip(tensor.shape,tensor.stride())):
                if stride == 0 and size > 1:
                    value = value.narrow(dim,0,1)
            # CUDA efficient-attention backward requires aligned attention-bias
            # strides. A CPU copy can compact this padded view and lose alignment.
            aligned_bias=(value.ndim==4 and value.shape[1]==1 and
                value.stride(-1)==1 and value.stride(-2)>value.shape[-1] and
                value.stride(-2)%8==0)
            return tensor.device, value.cpu(), tensor.shape, value.stride() if aligned_bias else None
        def unpack(saved):
            if isinstance(saved,torch.Tensor):
                return saved
            device,value,shape,strides = saved
            if strides is not None:
                restored=torch.empty_strided(value.shape,strides,dtype=value.dtype,device=device)
                restored.copy_(value)
            else:
                restored=value.to(device)
            return restored.expand(shape)
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            yield

    def configure_checkpointing(self, enabled):
        self.gradient_checkpointing = enabled
        if enabled:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()
        else:
            self.model.gradient_checkpointing_disable()
            self.model.eval()

    def parameters(self, role):
        return {name: value for name, value in self.model.named_parameters()
                if f".{role}." in name and "lora_" in name}

    def _restore_grad_flags(self):
        for name, value in self.model.named_parameters():
            value.requires_grad_("lora_" in name)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, view, role, params=None):
        if role != "reference":
            self.model.set_adapter(role)
        self._restore_grad_flags()
        context = self.model.disable_adapter() if role == "reference" else nullcontext()
        offload = (self.activation_offload()
                   if torch.is_grad_enabled() and
                      ((self.offload_teacher_activations and role == 'teacher') or
                       (self.offload_inner_activations and role == 'student'))
                   else nullcontext())
        with context, self.attention_context(role, params), offload:
            kwargs = dict(model_inputs(view), use_cache=False)
            mask = kwargs["attention_mask"]
            if mask.ndim == 2 and not bool(mask.all()):
                # Fully masked left-padding queries can produce NaN gradients in
                # eager attention. Give padding queries a harmless self edge;
                # real queries still cannot attend padding keys.
                length = mask.shape[1]
                causal = torch.ones(length, length, device=self.device, dtype=torch.bool).tril()
                allowed = causal[None, None] & mask[:, None, None, :].bool()
                padded_queries = ~mask[:, None, :, None].bool()
                allowed = allowed | (padded_queries & torch.eye(length, device=self.device, dtype=torch.bool)[None, None])
                # Pad physical row storage, then slice to the logical shape.
                # Out-of-place masked_fill would compact the view again.
                padded_length=(length+7)//8*8
                bias=torch.zeros((*allowed.shape[:-1],padded_length),device=self.device,
                    dtype=next(self.model.parameters()).dtype)[...,:length]
                bias.masked_fill_(~allowed,torch.finfo(bias.dtype).min)
                kwargs["attention_mask"] = bias
            if self.supports_selected_logits:
                kwargs["logits_to_keep"] = torch.arange(view["start"], view["start"] + view["length"], device=self.device)
            if params is None:
                result = self.model(**kwargs)
            else:
                result = functional_call(self.model, params, (), kwargs, strict=False)
        self._restore_grad_flags()
        return result.logits if self.supports_selected_logits else response_logits(result, view)


def build_model(config):
    settings = config["model"]
    if settings.get('offload_inner_activations'):
        from pathlib import Path
        meminfo = Path('/proc/meminfo')
        if meminfo.exists():
            values = dict(line.split(':',1) for line in meminfo.read_text().splitlines())
            available = int(values['MemAvailable'].split()[0]) * 1024 / 2**30
            minimum = settings.get('offload_min_host_gib',24)
            if available < minimum:
                raise RuntimeError(f'CPU activation offload needs at least {minimum} GiB available host RAM; currently {available:.1f} GiB. Retry after shared host memory is released.')
    dtype = getattr(torch, settings.get("dtype", "bfloat16"))
    base = AutoModelForCausalLM.from_pretrained(
        settings["name"], revision=settings.get("revision"), torch_dtype=dtype,
        attn_implementation=settings.get("attention_backend", "eager"),
    ).to(settings.get("device", "cuda"))
    expected = settings.get('expected_model_type')
    if expected and base.config.model_type != expected:
        raise ValueError(f"Expected {expected}, loaded {base.config.model_type}")
    lora = settings["lora"]
    bundle = ModelBundle(base, lora["rank"], lora["alpha"], lora["targets"])
    if config['experiment'].get('method') == 'meta_opsd':
        bundle.meta_gradient=settings.get('meta_gradient','unroll')
        bundle.offload_teacher_activations = settings.get('offload_teacher_activations', False)
        bundle.offload_inner_activations = settings.get('offload_inner_activations', False)
        # The mixed derivative traverses the inner STUDENT backward twice.
        # Teacher and Quiz forwards need only ordinary first-order backward.
        bundle.attention_overrides = {
            key: settings.get(field, settings.get('attention_backend','eager'))
            for key,field in [('teacher','teacher_attention_backend'),
                              ('reference','teacher_attention_backend'),('quiz','quiz_attention_backend')]}
    # Avoid mutating the shared backbone's dtype.
    for role in ("student", "teacher"):
        for parameter in bundle.parameters(role).values():
            parameter.data = parameter.data.float()
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings.get("revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return bundle, tokenizer

import torch
import time
from transformers import StoppingCriteria, StoppingCriteriaList


@torch.no_grad()
def sample_rollout(bundle, tokenizer, prompt_ids, max_new_tokens, temperature=0.7, top_p=0.95,
                   top_k=None, min_p=0.0, progress_callback=None):
    bundle.model.set_adapter("student")
    bundle._restore_grad_flags()
    inputs = torch.tensor([prompt_ids], device=bundle.device)
    kwargs = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens,
              "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": bundle.model.generation_config.eos_token_id or tokenizer.eos_token_id,
              "use_cache": True}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, min_p=min_p)
        if top_k is not None:
            kwargs["top_k"] = top_k
    if progress_callback is not None:
        width=len(prompt_ids)
        class Heartbeat(StoppingCriteria):
            def __init__(self): self.last=time.monotonic()
            def __call__(self,input_ids,scores,**unused):
                if time.monotonic()-self.last>=30:
                    progress_callback(input_ids.shape[1]-width)
                    self.last=time.monotonic()
                return False
        kwargs['stopping_criteria']=StoppingCriteriaList([Heartbeat()])
    was_training = bundle.model.training
    bundle.model.eval()
    output = bundle.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
    bundle.model.train(was_training)
    tokens = output[0, inputs.shape[1]:].tolist()
    eos = kwargs["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    return tokens, bool(tokens and tokens[-1] not in eos_ids and len(tokens) == max_new_tokens)

import torch
import torch.nn.functional as F
import pytest

from dual_opsd.data import make_batch_view, make_view
from dual_opsd.losses import forward_kl
from test_meta_gradient import tiny_bundle


def test_pointwise_clipping_matches_official_order_and_gradients():
    torch.manual_seed(4)
    t = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    s = torch.randn_like(t, requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    expected = F.kl_div(F.log_softmax(s/1.1, -1), F.log_softmax(t/1.1, -1),
                        reduction='none', log_target=True).clamp(max=0.05)[mask].sum()/mask.sum()
    actual = forward_kl(t, s, 1.1, token_clip=0.05, mask=mask, temperature_squared=False)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, (s, t), retain_graph=True, create_graph=True)
    e = torch.autograd.grad(expected, (s, t), retain_graph=True)
    for x, y in zip(a, e):
        torch.testing.assert_close(x, y)
    # Mixed derivative required by the teaching experiment remains available.
    assert torch.isfinite(torch.autograd.grad(a[0].square().sum(), t)[0]).all()


def test_batch_padding_and_accumulated_gradients_match():
    bundle = tiny_bundle()
    prompts, teacher_prompts, responses = [[1,2], [1,3,2]], [[1,7,2], [1,8,9,2]], [[4,5], [6,5,4]]
    student_view = make_batch_view(prompts, responses, 'cpu', 0)
    teacher_view = make_batch_view(teacher_prompts, responses, 'cpu', 0)
    params = tuple(bundle.parameters('student').values())
    with torch.no_grad():
        target = bundle.forward(teacher_view, 'reference')
    loss = forward_kl(target, bundle.forward(student_view, 'student'), 1.1,
                       token_clip=0.05, mask=student_view['response_mask'], temperature_squared=False)
    batched = torch.autograd.grad(loss, params)
    accumulated = [torch.zeros_like(p) for p in params]
    total = sum(map(len, responses))
    for s, t, r in zip(prompts, teacher_prompts, responses):
        with torch.no_grad():
            target = bundle.forward(make_view(t,r,'cpu'), 'reference')
        single = forward_kl(target, bundle.forward(make_view(s,r,'cpu'), 'student'), 1.1,
                             token_clip=0.05, temperature_squared=False)
        gradients = torch.autograd.grad(single, params)
        for result, gradient in zip(accumulated, gradients):
            result.add_(gradient * len(r) / total)
    for actual, expected in zip(batched, accumulated):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Flash attention check')
def test_meta_teacher_and_quiz_can_use_first_order_flash_attention():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from dual_opsd.models import ModelBundle
    from dual_opsd.meta import teacher_meta_grad
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    config._attn_implementation='eager'
    bundle=ModelBundle(Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16), rank=2, alpha=2)
    with torch.no_grad():
        for role in ('student','teacher'):
            for p in bundle.parameters(role).values():
                p.data=p.data.float()
                p.normal_(0,0.1)
    train=(make_view([1,2],[4,5],'cuda'),make_view([1,7,2],[4,5],'cuda'))
    quiz=(make_view([1,3],[6,5],'cuda'),make_view([1,8,3],[6,5],'cuda'))
    expected,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature_squared=False)
    bundle.attention_overrides={'teacher':'sdpa','reference':'sdpa','quiz':'sdpa'}
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature_squared=False)
    a=torch.cat([g.flatten() for g in actual])
    e=torch.cat([g.flatten() for g in expected])
    assert torch.isfinite(a).all() and a.norm()>0
    assert F.cosine_similarity(a,e,dim=0)>0.95
    bundle.offload_teacher_activations=True
    bundle.offload_inner_activations=True
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        offloaded,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature_squared=False)
    for actual_gradient, offloaded_gradient in zip(actual,offloaded):
        torch.testing.assert_close(actual_gradient,offloaded_gradient,atol=1e-8,rtol=1e-5)
    bundle.meta_gradient='jvp'
    bundle.model.config._attn_implementation='sdpa'
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        directional,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature_squared=False)
    directional=torch.cat([g.flatten() for g in directional])
    assert torch.isfinite(directional).all() and directional.norm()>0
    assert F.cosine_similarity(directional,e,dim=0)>0.95

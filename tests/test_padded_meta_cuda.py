import pytest
import torch

from dual_opsd.data import make_batch_view
from dual_opsd.models import ModelBundle
from dual_opsd.meta import accumulated_teacher_meta_grad


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA attention alignment')
def test_odd_padded_attention_bias_survives_activation_offload():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(23)
    config=Qwen3Config(vocab_size=32,hidden_size=128,intermediate_size=256,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=64)
    config._attn_implementation='sdpa'
    bundle=ModelBundle(Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16),rank=2,alpha=2)
    for role in ['student','teacher']:
        for p in bundle.parameters(role).values():
            p.data=p.data.float()
            with torch.no_grad():p.normal_(0,0.1)
    responses=[[4,5],[6,7,5,4]]
    train=(make_batch_view([[1,2],[1,3,2]],responses,'cuda',0),
           make_batch_view([[1,8,7,2],[1,9,8,7,2]],responses,'cuda',0))
    quiz=(make_batch_view([[1,3]],[[6,5]],'cuda',0),
          make_batch_view([[1,8,3]],[[6,5]],'cuda',0))
    bundle.offload_inner_activations=True
    bundle.offload_teacher_activations=True
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        gradients,_=accumulated_teacher_meta_grad(bundle,[train],quiz,0.3,
            temperature=1.1,token_clip=0.05,temperature_squared=False)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g)>0 for g in gradients)

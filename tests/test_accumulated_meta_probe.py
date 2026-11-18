import torch

from dual_opsd.data import make_batch_view
from dual_opsd.meta import teacher_meta_grad
from dual_opsd.meta import accumulated_teacher_meta_grad
from scripts.profile_dual_capacity import accumulated_inner_meta_grad
from test_meta_gradient import tiny_bundle


def test_shared_inner_accumulation_matches_full_batch_hypergradient():
    bundle=tiny_bundle()
    student=[[1,2],[1,3,2]]
    teacher=[[1,7,2],[1,8,9,2]]
    responses=[[4,5],[6,5,4]]
    def views(indices):
        return tuple(make_batch_view([prompts[i] for i in indices],
            [responses[i] for i in indices],'cpu',0) for prompts in [student,teacher])
    quiz=(make_batch_view([[1,3]],[[6,5]],'cpu',0),
          make_batch_view([[1,8,3]],[[6,5]],'cpu',0))
    initial={n:p.detach().clone() for n,p in bundle.model.named_parameters()}
    for clip in [None,0.0001]:
        options={'temperature':1.1,'token_clip':clip,'temperature_squared':False}
        expected,_=teacher_meta_grad(bundle,views([0,1]),quiz,0.3,**options)
        actual,_=accumulated_inner_meta_grad(bundle,[views([0]),views([1])],quiz,0.3,**options)
        integrated,_=accumulated_teacher_meta_grad(bundle,[views([0]),views([1])],quiz,0.3,**options)
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=1e-7,atol=1e-11)
        for a,e in zip(integrated,expected):
            torch.testing.assert_close(a,e,rtol=1e-7,atol=1e-11)
    for name,p in bundle.model.named_parameters():
        torch.testing.assert_close(p,initial[name],rtol=0,atol=0)

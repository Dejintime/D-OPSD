import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dual_opsd.data import make_view
from dual_opsd.losses import forward_kl
from dual_opsd.meta import teacher_meta_grad, virtual_update
from dual_opsd.models import ModelBundle


def tiny_bundle():
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=23, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        tie_word_embeddings=True, attention_dropout=0.0)
    config._attn_implementation = "eager"
    bundle = ModelBundle(Qwen2ForCausalLM(config), rank=2, alpha=2)
    bundle.model.double()
    # Nondegenerate adapter factors for numerical gradient verification.
    with torch.no_grad():
        for role in ("student", "teacher"):
            for value in bundle.parameters(role).values():
                value.normal_(0, 0.1)
    return bundle


def views():
    train = (make_view([1, 2], [4, 5], "cpu"), make_view([1, 7, 2], [4, 5], "cpu"))
    quiz = (make_view([1, 3], [6, 5], "cpu"), make_view([1, 8, 3], [6, 5], "cpu"))
    return train, quiz


def test_meta_directional_derivative_and_no_parameter_mutation():
    bundle = tiny_bundle()
    train, quiz = views()
    before = {name: p.detach().clone() for name, p in bundle.model.named_parameters()}
    gradients, _ = teacher_meta_grad(bundle, train, quiz, inner_lr=0.3)
    assert any(g.abs().max() > 1e-12 for g in gradients)
    for name, p in bundle.model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    teacher = tuple(bundle.parameters("teacher").values())
    direction = tuple(torch.randn_like(p) for p in teacher)
    predicted = sum((g * d).sum() for g, d in zip(gradients, direction)).item()

    def objective():
        with torch.no_grad():
            ref = bundle.forward(quiz[1], "reference")
        fast, _ = virtual_update(bundle, *train, inner_lr=0.3)
        return forward_kl(ref, bundle.forward(quiz[0], "student", fast)).item()

    original = [p.detach().clone() for p in teacher]
    eps = 1e-4
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            for p, start, d in zip(teacher, original, direction):
                p.copy_(start + sign * eps * d)
        values.append(objective())
    with torch.no_grad():
        for p, start in zip(teacher, original):
            p.copy_(start)
    finite_difference = (values[0] - values[1]) / (2 * eps)
    torch.testing.assert_close(torch.tensor(predicted), torch.tensor(finite_difference),
                               rtol=0.01, atol=1e-9)


def test_zero_inner_lr_has_zero_teacher_feedback():
    bundle = tiny_bundle()
    train, quiz = views()
    gradients, _ = teacher_meta_grad(bundle, train, quiz, inner_lr=0.0)
    assert all(torch.count_nonzero(g) == 0 for g in gradients)


def test_role_parameter_groups_are_disjoint_and_backbone_frozen():
    bundle = tiny_bundle()
    student = set(bundle.parameters("student"))
    teacher = set(bundle.parameters("teacher"))
    assert student and teacher and not student & teacher
    for name, parameter in bundle.model.named_parameters():
        assert parameter.requires_grad == (name in student or name in teacher)


def test_jvp_hypergradient_matches_full_unroll_with_and_without_clipping():
    bundle=tiny_bundle()
    train,quiz=views()
    for clip in (None,0.0001):
        bundle.meta_gradient='unroll'
        expected,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature=1.1,
                                     token_clip=clip,temperature_squared=False)
        bundle.meta_gradient='jvp'
        actual,_=teacher_meta_grad(bundle,train,quiz,0.3,temperature=1.1,
                                   token_clip=clip,temperature_squared=False)
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=1e-7,atol=1e-11)

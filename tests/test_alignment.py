from types import SimpleNamespace

import torch

from dual_opsd.data import make_view, response_logits
from dual_opsd.losses import forward_kl


def test_role_offsets_predict_identical_continuation_positions():
    student = make_view([1, 2], [6, 7, 0], "cpu")
    teacher = make_view([1, 3, 4, 2], [6, 7, 0], "cpu")
    assert student["start"] == 1 and teacher["start"] == 3
    for view in (student, teacher):
        logits = torch.zeros(1, view["input_ids"].shape[1], 8)
        for offset, token in enumerate([6, 7, 0]):
            logits[0, view["start"] + offset, token] = 1
        selected = response_logits(SimpleNamespace(logits=logits), view)
        assert selected.argmax(-1).tolist() == [[6, 7, 0]]
        assert selected.shape[1] == 3  # includes EOS, excludes all prompt tokens


def test_kl_direction_and_normalization():
    p = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]], dtype=torch.float64)
    q = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float64)
    expected = (p * (p.log() - q.log())).sum(-1).mean()
    torch.testing.assert_close(forward_kl(p.log(), q.log()), expected)
    torch.testing.assert_close(forward_kl(p.log(), p.log()), torch.zeros_like(expected))


def test_boxed_answer_formats():
    from scripts.prepare_data import extract_boxed
    assert extract_boxed(r"Answer $\boxed 2$.") == "2"
    assert extract_boxed(r"Answer $\boxed{\frac{1}{2}}$.") == r"\frac{1}{2}"
    assert extract_boxed(r"Answer $\fbox{9}$.") == "9"


def test_stratified_split_has_exact_count_and_no_overlap():
    import random
    from scripts.prepare_data import stratified_take
    rows = [{"id": i, "subject": str(i % 3), "level": i % 5} for i in range(101)]
    first, rest = stratified_take(rows, 60, random.Random(42))
    second, last = stratified_take(rest, 20, random.Random(42))
    assert [len(first), len(second), len(last)] == [60, 20, 21]
    ids = [[r["id"] for r in group] for group in (first, second, last)]
    assert len(set(sum(ids, []))) == 101

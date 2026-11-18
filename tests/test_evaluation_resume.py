import json
from types import SimpleNamespace

import pytest

import dual_opsd.evaluation as evaluation


def setup(monkeypatch):
    monkeypatch.setattr(evaluation,'student_prompt',lambda *args,**kwargs:[1,2])
    bundle=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100)))
    tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:r'\boxed{2}')
    rows=[{'id':'test/one','problem':'1+1','answer':'2'}]
    return bundle,tokenizer,rows


def test_resume_skips_completed_and_recovers_truncated_tail(monkeypatch,tmp_path):
    bundle,tokenizer,rows=setup(monkeypatch)
    calls=[]
    def sample(*args,**kwargs):
        calls.append(1)
        return [7],False
    monkeypatch.setattr(evaluation,'sample_rollout',sample)
    options={'samples_per_prompt':2,'temperature':1.0}
    summary=evaluation.evaluate_student(bundle,tokenizer,rows,tmp_path,8,generation_config=options)
    assert summary['evaluated_completions']==2 and summary['accuracy']==1
    with (tmp_path/'predictions.jsonl').open('ab') as stream: stream.write(b'{"partial":')
    evaluation.evaluate_student(bundle,tokenizer,rows,tmp_path,8,generation_config=options,resume=True)
    assert len(calls)==2
    assert len((tmp_path/'predictions.jsonl').read_text().splitlines())==2
    with pytest.raises(ValueError,match='configuration differs'):
        evaluation.evaluate_student(bundle,tokenizer,rows,tmp_path,9,generation_config=options,resume=True)


def test_interrupted_sample_is_retried_without_repeating_completed_sample(monkeypatch,tmp_path):
    bundle,tokenizer,rows=setup(monkeypatch)
    calls=[]
    def sample(*args,**kwargs):
        calls.append(1)
        if len(calls)==2: raise RuntimeError('simulated interruption')
        return [7],False
    monkeypatch.setattr(evaluation,'sample_rollout',sample)
    options={'samples_per_prompt':2,'temperature':1.0}
    with pytest.raises(RuntimeError,match='simulated interruption'):
        evaluation.evaluate_student(bundle,tokenizer,rows,tmp_path,8,generation_config=options)
    summary=evaluation.evaluate_student(bundle,tokenizer,rows,tmp_path,8,generation_config=options,resume=True)
    assert summary['evaluated_completions']==2 and len(calls)==3
    predictions=[json.loads(line) for line in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert [row['sample_index'] for row in predictions]==[0,1]


def test_official_extraction_uses_last_boxed_answer():
    assert evaluation.last_boxed_answer(r'\boxed{2} then \boxed{\frac{1}{3}}')==r'\frac{1}{3}'
    assert evaluation.last_boxed_answer(r'\boxed{2} then \boxed{oops') is None
    assert evaluation.last_boxed_answer('no final answer') is None

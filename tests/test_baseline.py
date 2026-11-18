import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from dual_opsd.baseline import baseline_config, read_schedule, compare_evaluations, audit_training
from dual_opsd.data import file_hash
from dual_opsd.trainer import Trainer
from test_meta_gradient import tiny_bundle


def test_config_preserves_matched_budget_and_reference_is_not_mutated():
    ref = yaml.safe_load(Path('configs/meta_qwen3_b32_100.yaml').read_text())
    original = copy.deepcopy(ref)
    result = baseline_config(ref)
    assert ref == original
    assert result == yaml.safe_load(Path('configs/opsd_qwen3_b32_100.yaml').read_text())
    assert result['train']['micro_batch_size'] * result['train']['gradient_accumulation_steps'] == 32
    assert result['rollout'] == ref['rollout']


def test_schedule_rejects_missing_ids_and_incomplete_runs(tmp_path):
    (tmp_path / 'training').mkdir()
    path = tmp_path / 'training/metrics.jsonl'
    cfg = {'train': {'max_steps': 2, 'micro_batch_size': 2, 'gradient_accumulation_steps': 1}}
    path.write_text(json.dumps({'step': 1, 'ids': ['a', 'b']}) + '\n')
    with pytest.raises(ValueError, match='every completed step'):
        read_schedule(tmp_path, cfg)
    path.write_text(path.read_text() + json.dumps({'step': 2, 'ids': ['c', 'd']}) + '\n')
    assert read_schedule(tmp_path, cfg)[1]['ids'] == ['c', 'd']


def test_fixed_teacher_replay_and_exact_checkpoint_resume(tmp_path, monkeypatch):
    from dual_opsd.data import make_batch_view
    from types import SimpleNamespace
    cfg = baseline_config(yaml.safe_load(Path('configs/meta_qwen3_b32_100.yaml').read_text()))
    cfg['data']['root'] = str(tmp_path)
    cfg['model']['name'] = 'RANDOM_TINY_TEST'
    cfg['train'].update(max_steps=2, micro_batch_size=2, gradient_accumulation_steps=2,
                        gradient_checkpointing=False, student_lr=0.01)
    rows = [{'id': str(i)} for i in range(4)]
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'distill.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    schedule = [{'step': 1, 'ids': ['3', '0', '2', '1']}, {'step': 2, 'ids': ['1', '2', '0', '3']}]
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(schedule))
    cfg['data'].update(distill_schedule_file=str(path), distill_schedule_sha256=file_hash(path))
    monkeypatch.setattr(Trainer, 'prepare', lambda self, e: (e, [1, 2], [1, 7, 2]))
    bundle = tiny_bundle()  # Teacher adapter is intentionally nonzero.
    trainer = Trainer(cfg, bundle, SimpleNamespace(pad_token_id=0), tmp_path / 'run')
    roles = []
    forward = bundle.forward
    def track(view, role, params=None):
        roles.append(role)
        return forward(view, role, params)
    monkeypatch.setattr(bundle, 'forward', track)
    def rollout(examples):
        responses = [[4, 5] if e[0]['id'] in ('0', '2') else [6, 5, 4] for e in examples]
        views = tuple(make_batch_view([e[i] for e in examples], responses, 'cpu', 0) for i in (1, 2))
        return views, {'ids': [e[0]['id'] for e in examples], 'truncated': 0.0,
                       'response_tokens': sum(map(len, responses))}
    monkeypatch.setattr(trainer, 'rollout_batch', rollout)
    before = {n: p.detach().clone() for n, p in bundle.model.named_parameters()}
    metrics = trainer.train_step()
    assert metrics['ids'] == schedule[0]['ids']
    assert 'reference' in roles and 'teacher' not in roles
    assert set(trainer.prepared) == {'distill'}
    assert not trainer.teacher_opt.state
    for n, p in bundle.model.named_parameters():
        if '.student.' not in n:
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    assert any(not torch.equal(p, before[n]) for n, p in trainer.student.items())
    trainer.save()
    expected = trainer.train_step()
    student = {n: p.detach().clone() for n, p in trainer.student.items()}
    trainer.resume(tmp_path / 'run/checkpoints/step_000001')
    resumed = trainer.train_step()
    assert resumed['ids'] == expected['ids'] == schedule[1]['ids']
    assert resumed['student_loss'] == expected['student_loss']
    for n, p in trainer.student.items():
        torch.testing.assert_close(p, student[n], rtol=0, atol=0)
    # A modified replay file must not silently affect a restarted run.
    path.write_text('[]')
    with pytest.raises(ValueError, match='hash mismatch'):
        Trainer(cfg, bundle, SimpleNamespace(pad_token_id=0), tmp_path / 'bad')


def test_comparison_checks_protocol_and_preserves_separate_initial_evaluations():
    # Minimal fixture with all identity fields checked by the comparison helper.
    keys = ('model dtype backend version data_sha256 ids thinking samples_per_prompt max_new_tokens '
            'temperature top_p top_k min_p presence_penalty stop_token_ids seed_policy prompt_style '
            'answer_extraction answer_markup').split()
    run = {'identity': {k: k for k in keys}, 'questions': 30, 'samples_per_question': 12,
           'baseline': {'evaluated_completions': 360, 'accuracy': 0.4},
           'trained': {'evaluated_completions': 360, 'accuracy': 0.45}}
    meta = {'parameters': {'samples_per_prompt': 12}, 'training_steps': 100,
            'training_batch_size': 32, 'datasets': {name: copy.deepcopy(run) for name in ('aime24', 'aime25')}}
    opsd = copy.deepcopy(meta)
    opsd['datasets']['aime25']['baseline']['accuracy'] = 0.41
    result = compare_evaluations(meta, opsd)
    assert result['dual_minus_opsd_percentage_points'] == 0
    assert result['initial_in_meta_eval_macro_average'] != result['initial_in_opsd_eval_macro_average']
    opsd['datasets']['aime25']['identity']['data_sha256'] = 'different'
    with pytest.raises(ValueError, match='identity mismatch'):
        compare_evaluations(meta, opsd)


def test_training_audit_uses_reference_ids_not_only_baseline_claims(tmp_path):
    reference, baseline = tmp_path / 'meta', tmp_path / 'opsd'
    config = yaml.safe_load(Path('configs/meta_qwen3_b32_100.yaml').read_text())
    config['train'].update(max_steps=2, micro_batch_size=1, gradient_accumulation_steps=1)
    for directory, cfg in ((reference, config), (baseline, baseline_config(config))):
        (directory / 'training').mkdir(parents=True)
        (directory / 'run_config.yaml').write_text(yaml.safe_dump(cfg))
        (directory / 'training/data_manifest.json').write_text('{}')
        (directory / 'training/coverage.json').write_text(json.dumps({'distill': {'kept': 2, 'ids': ['a', 'b']}}))
    schedule = [{'step': 1, 'ids': ['b']}, {'step': 2, 'ids': ['a']}]
    for directory in (reference, baseline):
        (directory / 'training/metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in schedule))
    path = baseline / 'distill_schedule.json'
    path.write_text(json.dumps(schedule))
    assert audit_training(reference, baseline)['distill_examples'] == 2
    path.write_text(json.dumps(list(reversed(schedule))))
    with pytest.raises(ValueError, match='schedule differs from reference'):
        audit_training(reference, baseline)

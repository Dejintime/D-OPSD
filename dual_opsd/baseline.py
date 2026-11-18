"""Matched fixed-teacher baseline preparation and audit (no model loading)."""
import copy
import json
from pathlib import Path
import yaml


def baseline_config(reference):
    if reference['experiment']['method'] != 'meta_opsd':
        raise ValueError('Expected a meta_opsd reference experiment')
    config = copy.deepcopy(reference)
    config['experiment'].update(name='opsd_qwen3_b32_100', method='opsd')
    # Checkpointing is a memory optimization for ordinary first-order training.
    config['train']['gradient_checkpointing'] = True
    for key in ('inner_steps', 'quiz_batch_size', 'inner_lr', 'teacher_lr', 'teacher_weight_decay'):
        config['train'].pop(key, None)
    for key in ('meta_gradient', 'teacher_attention_backend', 'quiz_attention_backend',
                'offload_teacher_activations', 'offload_inner_activations', 'offload_min_host_gib'):
        config['model'].pop(key, None)
    return config


def read_schedule(reference, config):
    rows = [json.loads(line) for line in (Path(reference) / 'training/metrics.jsonl').read_text().splitlines()]
    steps = config['train']['max_steps']
    size = config['train']['micro_batch_size'] * config['train']['gradient_accumulation_steps']
    if len(rows) != steps or any(r['step'] != i + 1 or len(r.get('ids', [])) != size
                                for i, r in enumerate(rows)):
        raise ValueError('Reference must contain every completed step and every distill ID')
    return [{'step': r['step'], 'ids': r['ids']} for r in rows]


def audit_training(reference, baseline):
    reference, baseline = Path(reference), Path(baseline)
    reference_config = yaml.safe_load((reference / 'run_config.yaml').read_text())
    expected_config = baseline_config(reference_config)
    actual_config = yaml.safe_load((baseline / 'run_config.yaml').read_text())
    for key in ('distill_schedule_file', 'distill_schedule_sha256'):
        actual_config['data'].pop(key, None)
    if actual_config != expected_config:
        raise ValueError('Baseline training settings do not match the reference budget and configuration')
    if json.loads((reference / 'training/data_manifest.json').read_text()) != json.loads(
            (baseline / 'training/data_manifest.json').read_text()):
        raise ValueError('Baseline data manifest differs from reference training')
    expected = read_schedule(reference, reference_config)
    if json.loads((baseline / 'distill_schedule.json').read_text()) != expected:
        raise ValueError('Baseline replay schedule differs from reference training')
    actual = [json.loads(line) for line in (baseline / 'training/metrics.jsonl').read_text().splitlines()]
    if [{'step': r['step'], 'ids': r['ids']} for r in actual] != expected:
        raise ValueError('Baseline did not use the full reference distill schedule')
    left = json.loads((reference / 'training/coverage.json').read_text())['distill']
    right = json.loads((baseline / 'training/coverage.json').read_text())['distill']
    if left != right:
        raise ValueError('Baseline and meta experiment have different eligible distill pools')
    return {'passed': True, 'updates': len(actual),
            'distill_examples': sum(len(row['ids']) for row in actual),
            'eligible_distill_examples': left['kept'], 'matching': 'student examples and update budget',
            'extra_meta_budget': 'Quiz rollouts and teacher hypergradient computation are not compute-matched'}


def compare_evaluations(meta, opsd):
    if meta['parameters'] != opsd['parameters']:
        raise ValueError('Evaluation sampling parameters differ')
    if any(r['training_steps'] != 100 or r['training_batch_size'] != 32 for r in (meta, opsd)):
        raise ValueError('Expected 100 updates with effective batch 32 for both methods')
    result = {'metric': 'Avg@12', 'training_steps': 100, 'effective_batch_size': 32,
              'parameters': meta['parameters'], 'datasets': {}}
    # Student adapter hashes must differ; every evaluation context below must match.
    identity_keys = ('model', 'dtype', 'backend', 'version', 'data_sha256', 'ids',
                     'thinking', 'samples_per_prompt', 'max_new_tokens', 'temperature',
                     'top_p', 'top_k', 'min_p', 'presence_penalty', 'stop_token_ids',
                     'seed_policy', 'prompt_style', 'answer_extraction', 'answer_markup')
    for name in ('aime24', 'aime25'):
        left, right = meta['datasets'][name], opsd['datasets'][name]
        if any(left['identity'][k] != right['identity'][k] for k in identity_keys):
            raise ValueError(f'Evaluation identity mismatch: {name}')
        for run in (left, right):
            if run['questions'] != 30 or run['samples_per_question'] != 12 or any(
                    run[role]['evaluated_completions'] != 360 for role in ('baseline', 'trained')):
                raise ValueError(f'Incomplete evaluation: {name}')
        result['datasets'][name] = {
            'initial_in_meta_eval': left['baseline'], 'initial_in_opsd_eval': right['baseline'],
            'opsd': right['trained'], 'dual_opsd': left['trained'],
            'dual_minus_opsd_percentage_points': 100 * (left['trained']['accuracy'] - right['trained']['accuracy'])}
    for role in ('initial_in_meta_eval', 'initial_in_opsd_eval', 'opsd', 'dual_opsd'):
        result[f'{role}_macro_average'] = sum(d[role]['accuracy'] for d in result['datasets'].values()) / 2
    result['dual_minus_opsd_percentage_points'] = 100 * (result['dual_opsd_macro_average'] - result['opsd_macro_average'])
    result['note'] = 'Sample/update matched; meta uses extra Quiz and second-order compute. Initial evaluations are retained separately.'
    return result

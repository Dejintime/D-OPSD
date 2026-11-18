"""Prepare/replay a matched OPSD baseline, optionally followed by full AIME eval."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_opsd.baseline import baseline_config, read_schedule, audit_training
from dual_opsd.data import file_hash
from dual_opsd.utils import load_config, save_json
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', help='Complete baseline checkpoint to resume')
    parser.add_argument('--evaluate', action='store_true', help='Run 1440-completion AIME suite after training')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if Path.cwd() != project:
        parser.error('Run from the project root so model and data paths resolve identically')
    reference, root = Path(args.reference_run).resolve(), Path(args.output_root).resolve()
    if root == reference:
        parser.error('Use a separate baseline output directory')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'baseline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = baseline_config(load_config(reference / 'run_config.yaml'))
    if config['train']['max_steps'] != 100 or config['train']['micro_batch_size'] * config['train']['gradient_accumulation_steps'] != 32:
        raise ValueError('This baseline is for the completed 100-step, batch-32 experiment')
    manifest = Path(config['data']['root']) / 'manifest.json'
    if json.loads(manifest.read_text()) != json.loads((reference / 'training/data_manifest.json').read_text()):
        raise ValueError('Current data manifest differs from reference training')
    schedule = read_schedule(reference, config)
    schedule_path = root / 'distill_schedule.json'
    if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
        raise ValueError('Output already belongs to a different distill schedule')
    save_json(schedule_path, schedule)
    config['data'].update(distill_schedule_file=str(schedule_path), distill_schedule_sha256=file_hash(schedule_path))
    config_path = root / 'run_config.yaml'
    if config_path.exists() and load_config(config_path) != config:
        raise ValueError('Output already belongs to a different training configuration')
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    save_json(root / 'baseline_plan.json', {'reference_run': str(reference), 'teacher': 'initial backbone, all adapters disabled',
        'config': str(config_path), 'reference_config_sha256': file_hash(reference / 'run_config.yaml'),
        'reference_metrics_sha256': file_hash(reference / 'training/metrics.jsonl'),
        'distill_schedule_sha256': file_hash(schedule_path), 'updates': 100, 'distill_examples': 3200,
        'quiz_examples': 0, 'sample_matched': True, 'compute_matched': False})
    command = [sys.executable, '-u', 'scripts/train.py', '--config', str(config_path), '--output', str(root / 'training')]
    if args.resume:
        command += ['--resume', args.resume]
    print(json.dumps({'training_command': command, 'prepared': True}), flush=True)
    if args.prepare_only:
        return
    final = root / 'training/checkpoints/step_000100/training_state.pt'
    if not final.exists():
        subprocess.run(command, check=True)
    save_json(root / 'training_audit.json', audit_training(reference, root))
    if args.evaluate:
        subprocess.run([sys.executable, '-u', 'scripts/run_aime_comparison.py', '--root', str(root), '--max-num-seqs', '8'], check=True)
        subprocess.run([sys.executable, 'scripts/compare_opsd_baseline.py', '--meta-root', str(reference), '--opsd-root', str(root)], check=True)


if __name__ == '__main__':
    main()

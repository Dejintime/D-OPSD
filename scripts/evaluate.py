import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_opsd.data import read_jsonl
from dual_opsd.data import file_hash
from dual_opsd.evaluation import evaluate_student
from dual_opsd.models import build_model
from dual_opsd.utils import load_checkpoint, load_config, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run")
    parser.add_argument("--checkpoint", help="Explicit checkpoint; otherwise latest numbered checkpoint")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, help="Debug only: labels output as subset evaluation")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Deterministic random subset; avoids a subject-sorted prefix")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.config and not args.run:
        parser.error("Provide --config or --run")
    config = load_config(args.config or str(Path(args.run) / "config.yaml"))
    seed_all(config["experiment"]["seed"])
    bundle, tokenizer = build_model(config)
    label = "initial"
    checkpoint = None
    if not args.initial:
        checkpoint = args.checkpoint
        if checkpoint is None:
            if not args.run:
                parser.error("Provide --checkpoint or --run when not using --initial")
            candidates = sorted(p for p in (Path(args.run) / "checkpoints").glob("step_*")
                                if (p / "training_state.pt").is_file())
            if not candidates:
                raise ValueError("No complete checkpoints")
            checkpoint = str(candidates[-1])
        load_checkpoint(checkpoint, bundle)
        label = Path(checkpoint).name
    rows = read_jsonl(Path(config["data"]["root"]) / f"{args.split}.jsonl")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        if args.limit > len(rows):
            parser.error("--limit exceeds the available split")
        rows = random.Random(args.sample_seed).sample(rows, args.limit)
        label += f"_subset_{len(rows)}_seed{args.sample_seed}"
    output = args.output or str(Path(args.run or config["logging"]["output_root"]) /
                               "evaluations" / f"{args.split}_{label}")
    if Path(output).exists() and not args.resume:
        raise ValueError("Refusing to overwrite an evaluation directory")
    summary = evaluate_student(bundle, tokenizer, rows, output,
                               config["evaluation"]["max_new_tokens"],
                               config.get("prompts"), config["evaluation"], resume=args.resume,
                               base_seed=config['experiment']['seed'], identity={
                                   'model':config['model'],
                                   'data_sha256':file_hash(Path(config['data']['root'])/f'{args.split}.jsonl'),
                                   'checkpoint':str(checkpoint) if checkpoint else None,
                                   'student_sha256':file_hash(Path(checkpoint)/'adapters/student/adapter_model.safetensors') if checkpoint else None})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

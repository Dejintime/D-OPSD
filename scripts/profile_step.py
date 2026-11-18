import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dual_opsd.models import build_model
from dual_opsd.trainer import Trainer
from dual_opsd.utils import load_config, save_json, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    if Path(args.output).exists():
        raise ValueError("Use a new output directory")
    config = load_config(args.config)
    seed_all(config["experiment"]["seed"])
    bundle, tokenizer = build_model(config)
    trainer = Trainer(config, bundle, tokenizer, args.output)
    for _ in range(args.warmup):
        trainer.train_step()
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows = [trainer.train_step() for _ in range(args.steps)]
    times = sorted(row["seconds"] for row in rows)
    report = {"steps": args.steps, "warmup": args.warmup, "mean_seconds": statistics.mean(times),
              "p50_seconds": statistics.median(times), "p90_seconds": times[min(len(times)-1, int(.9*len(times)))],
              "configured_training_steps": config['train']['max_steps'],
              "estimated_configured_steps_hours": statistics.mean(times) * config['train']['max_steps'] / 3600,
              "peak_allocated_gb": max(row.get("peak_allocated_gb", 0) for row in rows),
              "peak_reserved_gb": max(row.get("peak_reserved_gb", 0) for row in rows),
              "note": "Full updating cycles, including train and quiz generation; estimate assumes this length distribution"}
    save_json(Path(args.output) / "profile.json", report)
    from dual_opsd.data import write_jsonl
    write_jsonl(Path(args.output) / "profile_steps.jsonl", rows)
    if args.save_checkpoint:
        trainer.save()
    print(report)


if __name__ == "__main__":
    main()

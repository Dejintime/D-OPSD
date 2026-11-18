"""A paired developer-set pilot; never uses the final test set for tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server_debug.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=50)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_limit < 1:
        parser.error("steps and eval-limit must be positive")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    if config['evaluation'].get('samples_per_prompt',1) != 1:
        raise ValueError('Paired pilot expects one prediction per question; use evaluate.py for official 12-sample evaluation')
    config["experiment"]["name"] = root.name
    config["train"]["max_steps"] = args.steps
    cfg_path = root / "pilot_config.yaml"
    if cfg_path.exists() and yaml.safe_load(cfg_path.read_text()) != config:
        raise ValueError("Existing pilot has a different config")
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    started = time.time()
    def stage(name, command):
        print(f"START {name}: {' '.join(command)}", flush=True)
        (root / "status.json").write_text(json.dumps({"stage": name, "status": "running"}))
        with (root / f"{name}.log").open("a") as log:
            process = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            (root / "status.json").write_text(json.dumps({"stage": name, "status": "failed", "exit_code": process.returncode}))
            raise RuntimeError(f"{name} failed; see {root / (name + '.log')}")
        print(f"DONE {name}", flush=True)
    if not (root / "baseline/summary.json").exists():
        stage("baseline", ["scripts/evaluate.py", "--config", str(cfg_path), "--initial",
                           "--split", "dev", "--limit", str(args.eval_limit), "--output", str(root / "baseline")])
    train_command = ["scripts/train.py", "--config", str(cfg_path), "--output", str(root / "training")]
    checkpoints = sorted((root / "training/checkpoints").glob("step_*/training_state.pt"))
    final = root / f"training/checkpoints/step_{args.steps:06d}/training_state.pt"
    if not final.exists():
        if checkpoints:
            train_command += ["--resume", str(checkpoints[-1].parent)]
        stage("training", train_command)
    if not (root / "trained/summary.json").exists():
        stage("trained", ["scripts/evaluate.py", "--run", str(root / "training"),
                          "--checkpoint", str(final.parent), "--split", "dev", "--limit", str(args.eval_limit),
                          "--output", str(root / "trained")])
    load = lambda path: json.loads(path.read_text())
    baseline, trained = load(root / "baseline/summary.json"), load(root / "trained/summary.json")
    rows = lambda path: {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines())}
    before, after = rows(root / "baseline/predictions.jsonl"), rows(root / "trained/predictions.jsonl")
    if before.keys() != after.keys() or len(before) != args.eval_limit:
        raise ValueError("Pilot evaluations must have the same requested examples")
    comparison = {"kind": "small_developer_set_pilot_not_final_benchmark", "steps": args.steps,
                  "baseline": baseline, "trained": trained,
                  "accuracy_delta_percentage_points": 100 * (trained["accuracy"] - baseline["accuracy"]),
                  "improved_ids": [key for key in before if not before[key]["correct"] and after[key]["correct"]],
                  "regressed_ids": [key for key in before if before[key]["correct"] and not after[key]["correct"]],
                  "wall_seconds_this_invocation": time.time() - started}
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    (root / "status.json").write_text(json.dumps({"stage": "complete", "status": "complete"}))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_opsd.models import build_model
from dual_opsd.trainer import Trainer
from dual_opsd.utils import load_config, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_all(config["experiment"]["seed"])
    bundle, tokenizer = build_model(config)
    output = args.output or str(Path(config["logging"]["output_root"]) /
                               (config["experiment"]["name"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    if Path(output, "config.yaml").exists() and not args.resume:
        raise ValueError("Output already has a run; use a new directory or --resume")
    if args.resume and Path(output, "config.yaml").exists():
        if load_config(Path(output, "config.yaml")) != config:
            raise ValueError("Resume config differs from existing run")
    trainer = Trainer(config, bundle, tokenizer, output)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()

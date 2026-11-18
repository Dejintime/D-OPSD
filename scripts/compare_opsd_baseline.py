"""Compare complete, identically decoded OPSD and dual OPSD AIME suites."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_opsd.baseline import audit_training, compare_evaluations
from dual_opsd.utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-root', required=True)
    parser.add_argument('--opsd-root', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    meta, opsd = Path(args.meta_root), Path(args.opsd_root)
    audit = audit_training(meta, opsd)
    def read(root):
        return json.loads((root / 'aime24_aime25_38912_s12/comparison.json').read_text())
    result = compare_evaluations(read(meta), read(opsd))
    result.update(training_audit=audit, meta_root=str(meta.resolve()), opsd_root=str(opsd.resolve()))
    output = Path(args.output) if args.output else opsd / 'opsd_vs_dual_comparison.json'
    save_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

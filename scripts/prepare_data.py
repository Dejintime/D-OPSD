"""Prepare immutable problem-level splits. Run from the repository root."""
import argparse
from collections import defaultdict
import random
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dual_opsd.data import file_hash, read_jsonl, write_jsonl
from dual_opsd.utils import save_json


def extract_boxed(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*", text))
    if not matches:
        raise ValueError("MATH solution lacks boxed answer")
    start = matches[-1].end()
    if start == len(text):
        raise ValueError("Empty boxed answer")
    if text[start] != "{":
        token = re.match(r"[-+]?\d+(?:\.\d+)?|\\[A-Za-z]+|[^\s$]", text[start:])
        if token:
            return token.group()
        raise ValueError("Unsupported unbraced boxed answer")
    start += 1
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    raise ValueError("Unbalanced boxed answer")


def stratified_take(rows, count, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subject"], row["level"])].append(row)
    quotas = {key: count * len(group) // len(rows) for key, group in groups.items()}
    keys = sorted(groups, key=lambda key: (-(count * len(groups[key]) % len(rows)), key))
    for key in keys[:count - sum(quotas.values())]:
        quotas[key] += 1
    taken, remaining = [], []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        taken.extend(groups[key][:quotas[key]])
        remaining.extend(groups[key][quotas[key]:])
    rng.shuffle(taken)
    return taken, remaining


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["math", "gsm8k"], default="math")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--revision", required=True, help="Pinned source revision or commit")
    parser.add_argument("--test-revision", help="Pinned MATH-500 revision")
    parser.add_argument("--math500-file", help="Optional local copy of the pinned MATH-500 test.jsonl")
    args = parser.parse_args()
    root = Path(args.output or f"data/processed/{args.dataset}")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to overwrite an existing processed dataset")
    rng = random.Random(args.seed)
    train, test = [], []
    if args.dataset == "math":
        if not args.test_revision:
            parser.error("MATH requires --test-revision for MATH-500")
        subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                    "number_theory", "prealgebra", "precalculus"]
        for subject in subjects:
            source = load_dataset("EleutherAI/hendrycks_math", subject, revision=args.revision)
            for index, row in enumerate(source["train"]):
                train.append({"id": f"math/train/{subject}/{index}", "problem": row["problem"],
                              "solution": row["solution"], "answer": extract_boxed(row["solution"]),
                              "subject": subject,
                              "level": int(row["level"].split()[-1]) if row["level"].split()[-1].isdigit() else 0,
                              "source_split": "train"})
        source = (read_jsonl(args.math500_file) if args.math500_file else
                  load_dataset("HuggingFaceH4/MATH-500", split="test", revision=args.test_revision))
        if len(source) != 500:
            raise ValueError("MATH-500 must contain exactly 500 problems")
        for row in source:
            test.append({"id": row["unique_id"], "problem": row["problem"],
                         "solution": row["solution"], "answer": row["answer"],
                         "subject": row["subject"], "level": row["level"], "source_split": "test"})
        expected, dev_count = 7500, 500
    else:
        source = load_dataset("openai/gsm8k", "main", revision=args.revision)
        for split in ("train", "test"):
            for index, row in enumerate(source[split]):
                (train if split == "train" else test).append({
                    "id": f"gsm8k/{split}/{index}", "problem": row["question"],
                    "solution": row["answer"].split("####")[0].strip(),
                    "answer": row["answer"].split("####")[-1].strip(),
                    "subject": "arithmetic", "level": 0, "source_split": split})
        expected, dev_count = 7473, 473
    if len(train) != expected:
        raise ValueError(f"Unexpected train count: {len(train)}")
    # Preserve the entire standard test set; remove train overlap and repeated train questions.
    key_of = lambda row: re.sub(r"\s+", "", row["problem"])
    seen = {key_of(row) for row in test}
    unique_train, removed = [], []
    for row in train:
        key = key_of(row)
        if key in seen:
            removed.append(row["id"])
        else:
            seen.add(key)
            unique_train.append(row)
    distill, rest = stratified_take(unique_train, len(unique_train) - 1000 - dev_count, rng)
    quiz, dev = stratified_take(rest, 1000, rng)
    assert len(dev) == dev_count
    splits = {"distill": distill, "quiz": quiz, "dev": dev, "test": test}
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(root / f"{split}.jsonl", rows)
    save_json(root / "manifest.json", {"dataset": args.dataset, "revision": args.revision,
              "test_revision": args.test_revision, "seed": args.seed,
              "local_test_source_sha256": file_hash(args.math500_file) if args.math500_file else None,
              "removed_duplicate_train_ids": removed,
              "unknown_difficulty_count": sum(r["level"] == 0 for r in unique_train) if args.dataset == "math" else 0,
              "splits": {key: {"count": len(rows), "sha256": file_hash(root / f"{key}.jsonl")}
                         for key, rows in splits.items()}})
    print(root)


if __name__ == "__main__":
    main()

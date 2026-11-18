"""Prepare official OPSD source data with independent research holdouts."""
import argparse
import hashlib
import random
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from dual_opsd.data import file_hash, read_jsonl, write_jsonl
from dual_opsd.utils import save_json

REVISION = '1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw/openthoughts')
    parser.add_argument('--output', default='data/processed/openthoughts')
    parser.add_argument('--test-file', default='data/processed/math/test.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiz-size', type=int, default=1000)
    parser.add_argument('--dev-size', type=int, default=500)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite existing prepared data')
    files = sorted(Path(args.raw).glob('train-*.parquet'))
    if len(files) != 2:
        raise ValueError('Expected two complete Parquet shards')
    source = []
    for file in files:
        rows = pq.read_table(file, columns=['source', 'problem', 'solution', 'Answer']).to_pylist()
        for i, row in enumerate(rows):
            source.append({'id': f'openthoughts/{file.stem}/{i}', 'problem': row['problem'],
                'solution': row['solution'], 'answer': row['Answer'] or '',
                'subject': row['source'], 'level': 0, 'source_split': 'train'})
    if len(source) != 29434:
        raise ValueError(f'Unexpected pinned dataset size: {len(source)}')
    test = read_jsonl(args.test_file)
    normalize = lambda text: re.sub(r'\s+', '', text)
    blocked = {normalize(row['problem']) for row in test}
    seen, cleaned, removed = set(), [], []
    for row in source:
        key = normalize(row['problem'])
        reason = 'test_overlap' if key in blocked else ('duplicate' if key in seen else None)
        if reason:
            removed.append({'id': row['id'], 'reason': reason})
        else:
            seen.add(key)
            cleaned.append(row)
    # Gold answers are unnecessary for distillation/Quiz; development rows need gold.
    candidates = [r for r in cleaned if r['answer'].strip()]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if len(candidates) < args.dev_size:
        raise ValueError('Not enough gold-answer examples for development')
    dev = candidates[:args.dev_size]
    dev_ids = {r['id'] for r in dev}
    rest = [r for r in cleaned if r['id'] not in dev_ids]
    rng.shuffle(rest)
    quiz, distill = rest[:args.quiz_size], rest[args.quiz_size:]
    splits = {'distill': distill, 'quiz': quiz, 'dev': dev, 'test': test}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(output / f'{name}.jsonl', rows)
    manifest = {'dataset': 'siyanzhao/Openthoughts_math_30k_opsd', 'revision': REVISION,
        'raw_count': len(source), 'seed': args.seed, 'removed': removed,
        'notes': 'Research split for dual OPSD: independent Quiz/dev; exact normalized MATH-500 overlaps removed. Not the unmodified official full train split.',
        'source_files': {p.name: file_hash(p) for p in files},
        'splits': {name: {'count': len(rows), 'sha256': file_hash(output / f'{name}.jsonl')}
                   for name, rows in splits.items()}}
    save_json(output / 'manifest.json', manifest)
    print({name: len(rows) for name, rows in splits.items()})
    print('removed', len(removed))


if __name__ == '__main__':
    main()

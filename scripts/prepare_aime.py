"""Prepare the pinned AIME24/25 datasets used by the official OPSD evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import pyarrow.parquet as pq


SOURCES={
    'aime24':{
        'repo':'HuggingFaceH4/aime_2024',
        'revision':'2fe88a2f1091d5048c0f36abc874fb997b3dd99a',
        'file':'data/train-00000-of-00001.parquet',
        'sha256':'26139847601a5037c237d5928b195e7260ca8074cf4f264b794af42847f79ccf',
    },
    'aime25':{
        'repo':'yentinglin/aime_2025',
        'revision':'6f71d77b0b89b9dabe07ab466c51df33f514df7f',
        'file':'data/train-00000-of-00001-243207c6c994e1bd.parquet',
        'sha256':'b74b507c1a4de7c9fafd2c83dfe2ef7b80579bcf0e0a754a1765db0f21c205fc',
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',default='data/raw/aime')
    parser.add_argument('--output',default='data/processed/aime')
    args=parser.parse_args()
    raw=Path(args.raw)
    output=Path(args.output)
    raw.mkdir(parents=True,exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite an existing prepared dataset')
    prepared={}
    manifest={'official_opsd_commit':'ae7d2519e94920c4eb6206c0c26de46d9c50abae','datasets':{}}
    seen=set()
    for name,source in SOURCES.items():
        path=raw/f'{name}_{source["revision"]}.parquet'
        url=f'https://huggingface.co/datasets/{source["repo"]}/resolve/{source["revision"]}/{source["file"]}'
        if not path.exists():
            with urllib.request.urlopen(url,timeout=60) as response:
                content=response.read()
            temporary=path.with_suffix('.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
        if sha256(path)!=source['sha256']:
            raise ValueError(f'Raw dataset hash mismatch: {name}')
        rows=pq.read_table(path).to_pylist()
        if len(rows)!=30:
            raise ValueError(f'Expected all 30 questions: {name}')
        converted=[]
        for index,row in enumerate(rows):
            problem=row['problem'].strip()
            answer=str(row['answer']).strip()
            normalized=re.sub(r'\s+','',problem)
            if not problem or normalized in seen or not re.fullmatch(r'\d{1,3}',answer):
                raise ValueError(f'Invalid or duplicate question: {name}/{index}')
            seen.add(normalized)
            converted.append({'id':f'{name}/{index:02d}','benchmark':name.upper(),
                'source_id':row['id'],'problem':problem,'answer':answer,
                'solution':row.get('solution') or '', 'source_split':'train'})
        prepared[name]=converted
        manifest['datasets'][name]={**source,'split':'train','count':30,'raw_file':str(path),
            'data_file':str(output/f'{name}.jsonl')}
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in prepared.items():
        path=output/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        manifest['datasets'][name]['processed_sha256']=sha256(path)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

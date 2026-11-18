import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from .prompts import student_prompt
from .rollout import sample_rollout
from .utils import save_json, seed_all


def math_answer_markup(value):
    """Wrap bare LaTeX answers, including escaped currency dollars, in math mode."""
    value=str(value)
    return value if re.search(r'(?<!\\)\$',value) else '$'+value+'$'


def last_boxed_answer(text):
    start=text.rfind('\\boxed')
    if start<0 or not text.startswith('\\boxed{',start): return None
    depth=1
    for index in range(start+7,len(text)):
        depth+=(text[index]=='{')-(text[index]=='}')
        if depth==0: return text[start+7:index].strip()
    return None


def evaluate_student(bundle, tokenizer, rows, output_dir, max_new_tokens=1024,
                     prompt_config=None, generation_config=None, resume=False,
                     base_seed=42, identity=None):
    try:
        from math_verify import parse, verify, LatexExtractionConfig
    except ImportError as error:
        raise RuntimeError('Install math-verify before evaluation') from error
    started=time.perf_counter()
    prompt_config=prompt_config or {}
    generation_config=generation_config or {}
    samples=int(generation_config.get('samples_per_prompt',1))
    if samples<1 or not rows: raise ValueError('Empty evaluation or invalid sample count')
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    metadata={'ids':[r['id'] for r in rows], 'gold':[r['answer'] for r in rows],
              'samples_per_prompt':samples, 'max_new_tokens':max_new_tokens,
              'prompts':prompt_config, 'generation':generation_config, 'base_seed':base_seed,
              'seed_policy':'sha256(base_seed, question_id, sample_index), first 32 bits',
              'identity':identity or {}}
    meta_path=output/'evaluation_metadata.json'
    if meta_path.exists():
        if not resume: raise ValueError('Existing evaluation; explicitly request resume')
        if json.loads(meta_path.read_text())!=metadata:
            raise ValueError('Evaluation identity/configuration differs from saved metadata')
    else:
        save_json(meta_path,metadata)
    predictions=[]
    path=output/'predictions.jsonl'
    if path.exists():
        raw=path.read_bytes()
        valid_end=0
        for line in raw.splitlines(keepends=True):
            try:
                row=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if valid_end+len(line)!=len(raw): raise ValueError('Corrupt prediction journal')
                break
            predictions.append(row)
            valid_end+=len(line)
        if valid_end!=len(raw):
            with path.open('r+b') as stream: stream.truncate(valid_end)
        if valid_end and raw[valid_end-1:valid_end]!=b'\n':
            with path.open('ab') as stream: stream.write(b'\n')
    done={(r['id'],r['sample_index']) for r in predictions}
    expected={(r['id'],j) for r in rows for j in range(samples)}
    if len(done)!=len(predictions) or not done<=expected:
        raise ValueError('Duplicate/unexpected prediction in journal')
    # Preflight every gold and context before spending GPU time.
    prepared={}
    for row in rows:
        prompt=student_prompt(tokenizer,row['problem'],generation_config.get('prompt_style',prompt_config.get('style','legacy')),
                              generation_config.get('thinking',False))
        if len(prompt)+max_new_tokens>bundle.model.config.max_position_embeddings:
            raise ValueError(f"Evaluation context exceeds model window: {row['id']}")
        official=generation_config.get('answer_extraction')=='last_boxed'
        gold_text=str(row['answer'])
        gold=(parse(math_answer_markup(gold_text),fallback_mode='no_fallback')
              if official else parse('\\boxed{'+gold_text+'}',extraction_config=[LatexExtractionConfig()]))
        if not gold: raise ValueError(f"Unparseable gold answer: {row['id']}")
        prepared[row['id']]=(prompt,gold)
    total=len(rows)*samples
    def progress(current=None):
        value={'completed':len(predictions),'total':total,
               'correct_so_far':sum(r['correct'] for r in predictions),
               'generated_tokens_completed':sum(r['tokens'] for r in predictions),
               'wall_seconds_this_invocation':time.perf_counter()-started,
               'current':current}
        temporary=output/'progress.tmp'
        temporary.write_text(json.dumps(value))
        temporary.replace(output/'progress.json')
    progress()
    with path.open('a',encoding='utf-8') as stream:
        for row in rows:
            prompt,gold=prepared[row['id']]
            for sample_index in range(samples):
                if (row['id'],sample_index) in done: continue
                key=f"{base_seed}:{row['id']}:{sample_index}".encode()
                sample_seed=int.from_bytes(hashlib.sha256(key).digest()[:4],'big')
                seed_all(sample_seed)
                current={'id':row['id'],'sample_index':sample_index,'generated_tokens':0}
                progress(current)
                def heartbeat(count):
                    current['generated_tokens']=count
                    progress(current)
                sample_started=time.perf_counter()
                tokens,truncated=sample_rollout(bundle,tokenizer,prompt,max_new_tokens,
                    temperature=generation_config.get('temperature',0),
                    top_p=generation_config.get('top_p',0.95),
                    top_k=generation_config.get('top_k',0),progress_callback=heartbeat)
                text=tokenizer.decode(tokens,skip_special_tokens=True)
                boxed=last_boxed_answer(text) if official else None
                fallback=False
                try:
                    predicted=(parse(math_answer_markup(boxed),fallback_mode='no_fallback')
                               if official and boxed is not None else
                               ([] if official else parse(text,extraction_config=[LatexExtractionConfig()])))
                    correct=bool(predicted and verify(gold,predicted))
                except Exception:
                    predicted=[]
                    normalize=lambda x:x.replace('$','').replace(' ','').lower().strip()
                    correct=bool(official and boxed is not None and normalize(boxed)==normalize(str(row['answer'])))
                    fallback=True
                result={'id':row['id'],'sample_index':sample_index,'sample_seed':sample_seed,
                    'prediction':text,'answer':row['answer'],'extracted':[str(x) for x in predicted],
                    'correct':correct,'boxed_answer':boxed,'grader_exception_fallback':fallback,
                    'extraction_failed':not bool(predicted),'truncated':truncated,'tokens':len(tokens),
                    'seconds':time.perf_counter()-sample_started}
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                predictions.append(result)
                done.add((row['id'],sample_index))
                progress()
                print(f"{len(predictions)}/{total} correct={sum(r['correct'] for r in predictions)} tokens={len(tokens)}",flush=True)
    summary={'n':len(rows),'samples_per_prompt':samples,'evaluated_completions':len(predictions),
        'metric':'Avg@'+str(samples) if samples>1 else 'accuracy',
        'accuracy':sum(r['correct'] for r in predictions)/len(predictions),
        'extraction_failure_rate':sum(r['extraction_failed'] for r in predictions)/len(predictions),
        'truncation_rate':sum(r['truncated'] for r in predictions)/len(predictions),
        'generated_tokens':sum(r['tokens'] for r in predictions),
        'seconds':sum(r.get('seconds',0) for r in predictions),
        'decoding':'greedy' if generation_config.get('temperature',0)==0 else 'sampling',
        'thinking':generation_config.get('thinking',False), 'seed_policy':metadata['seed_policy']}
    save_json(output/'summary.json',summary)
    return summary

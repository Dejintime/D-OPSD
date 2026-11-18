"""Resumable before/after math evaluation with a shared vLLM engine."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''): sha.update(data)
    return sha.hexdigest()


def read_journal(path):
    if not path.exists(): return []
    raw=path.read_bytes()
    rows=[]
    end=0
    for line in raw.splitlines(keepends=True):
        try: row=json.loads(line)
        except ValueError:
            if end+len(line)!=len(raw): raise ValueError('Corrupt journal')
            break
        rows.append(row)
        end+=len(line)
    if end!=len(raw):
        with path.open('r+b') as stream: stream.truncate(end)
    if end and raw[end-1:end]!=b'\n':
        with path.open('ab') as stream: stream.write(b'\n')
    return rows


def wait_for_resources(training_root,final,root):
    while not (final/'training_state.pt').is_file():
        pid=int((training_root/'training.pid').read_text())
        try: command=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError: command=b''
        if b'scripts/train.py' not in command:
            raise RuntimeError('Training ended before step 100; inspect training.log')
        atomic_json(root/'vllm_state.json',{'status':'waiting_for_step_100','updated':time.time()})
        time.sleep(10)
    while True:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=15,check=True)
        free=int(result.stdout.strip().splitlines()[0])
        if free>=22528: return
        atomic_json(root/'vllm_state.json',{'status':'waiting_for_gpu','free_mib':free,'updated':time.time()})
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output-root',help='Separate evaluation directory; defaults to the training run')
    parser.add_argument('--data-file',help='Prepared evaluation JSONL; defaults to the training data dev split')
    parser.add_argument('--dataset-label',default='held-out OpenThoughts developer questions')
    parser.add_argument('--limit',type=int,default=50,help='Seed-42 random subset; 0 evaluates every row in file order')
    parser.add_argument('--max-new-tokens',type=int,default=38912)
    parser.add_argument('--samples-per-prompt',type=int,default=12)
    parser.add_argument('--max-model-len',type=int,default=40960)
    parser.add_argument('--compat',action='store_true')
    parser.add_argument('--compat-adapter',default='outputs/qwen3_meta_validation_v1/checkpoints/step_000001/adapters/student')
    parser.add_argument('--max-num-seqs',type=int,default=8)
    args=parser.parse_args()
    if args.limit<0 or min(args.max_new_tokens,args.samples_per_prompt,args.max_model_len,args.max_num_seqs)<1:
        parser.error('Invalid evaluation budget')
    training_root=Path(args.root).resolve()
    root=Path(args.output_root).resolve() if args.output_root else training_root
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'vllm_pair.lock').open('a')
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('A vLLM comparison process already owns this run') from error
    final=training_root/'training/checkpoints/step_000100'
    if not args.compat: wait_for_resources(training_root,final,root)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.lora.request import LoRARequest
    import vllm
    import yaml
    from transformers import AutoTokenizer
    from math_verify import parse,verify
    from dual_opsd.prompts import student_prompt
    from dual_opsd.evaluation import last_boxed_answer,math_answer_markup

    config=yaml.safe_load((training_root/'run_config.yaml').read_text())
    model=str(Path(config['model']['name']).resolve())
    adapter=str(Path(args.compat_adapter).resolve()) if args.compat else str(final/'adapters/student')
    tokenizer=AutoTokenizer.from_pretrained(model)
    max_tokens=8 if args.compat else args.max_new_tokens
    context=128 if args.compat else args.max_model_len
    dataset_path=Path(args.data_file) if args.data_file else Path(config['data']['root'])/'dev.jsonl'
    all_rows=[json.loads(line) for line in dataset_path.read_text().splitlines()]
    rows=([{'id':'compat','problem':'What is 1+1?','answer':'2'}] if args.compat else
          all_rows if args.limit==0 else random.Random(42).sample(all_rows,args.limit))
    samples=1 if args.compat else args.samples_per_prompt
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Evaluation dataset is empty or contains duplicate IDs')
    golds={}
    prompts={}
    for row in rows:
        value=str(row['answer'])
        golds[row['id']]=parse(math_answer_markup(value),fallback_mode='no_fallback')
        if not golds[row['id']]: raise ValueError(f"Unparseable gold: {row['id']}")
        prompts[row['id']]=student_prompt(tokenizer,row['problem'],'opsd_eval',True)
        if len(prompts[row['id']])+max_tokens>context:
            raise ValueError(f"Context budget exceeded: {row['id']}")
    if not args.compat:
        atomic_json(root/'evaluation_preflight.json',{'passed':True,'dataset':args.dataset_label,
            'data_file':str(dataset_path.resolve()),'data_sha256':digest(dataset_path),
            'questions':len(rows),'samples_per_prompt':samples,'total_completions':len(rows)*samples*2,
            'max_new_tokens':max_tokens,'max_prompt_tokens':max(map(len,prompts.values())),
            'model_context':context,'ids':[r['id'] for r in rows]})
        atomic_json(root/'vllm_state.json',{'status':'initializing_engine','updated_unix':time.time()})
    llm=LLM(model=model,dtype='bfloat16',tensor_parallel_size=1,trust_remote_code=False,
        max_model_len=context,gpu_memory_utilization=0.30 if args.compat else 0.90,
        max_num_seqs=2 if args.compat else args.max_num_seqs,
        max_num_batched_tokens=128 if args.compat else 8192,
        enable_chunked_prefill=True,enable_prefix_caching=True,
        enforce_eager=True,enable_lora=True,max_lora_rank=64,max_loras=1,max_cpu_loras=1,
        distributed_executor_backend='mp',generation_config='vllm',seed=42)
    lora=LoRARequest('student_step100' if not args.compat else 'compat_student',1,adapter)
    engine=llm.llm_engine
    eos=json.loads(Path(model,'generation_config.json').read_text()).get('eos_token_id',tokenizer.eos_token_id)
    eos=eos if isinstance(eos,list) else [eos]
    outputs={name:root/name for name in ['baseline_vllm','trained_vllm']}
    records={phase:[] for phase in outputs}
    identity={'backend':'vllm','version':vllm.__version__,'model':model,'dtype':'bfloat16',
        'student_adapter_sha256':digest(Path(adapter)/'adapter_model.safetensors'),
        'dataset':args.dataset_label,'data_file':str(dataset_path.resolve()),
        'data_sha256':digest(dataset_path),'ids':[r['id'] for r in rows],
        'thinking':True,'samples_per_prompt':samples,'max_new_tokens':max_tokens,
        'temperature':1.0,'top_p':0.95,'top_k':-1,'min_p':0.0,'presence_penalty':0.0,
        'stop_token_ids':eos,'seed_policy':'sha256(42,question_id,sample_index), first 32 bits',
        'prompt_style':'opsd_eval','answer_extraction':'last_boxed',
        'answer_markup':'wrap_bare_latex_including_escaped_currency'}
    if not args.compat:
        for phase,directory in outputs.items():
            directory.mkdir(exist_ok=True)
            meta=directory/'evaluation_metadata.json'
            expected={**identity,'phase':phase}
            if meta.exists() and json.loads(meta.read_text())!=expected:
                raise ValueError('Saved vLLM evaluation differs from current configuration')
            atomic_json(meta,expected)
            records[phase]=read_journal(directory/'predictions.jsonl')
    done={phase:{(r['id'],r['sample_index']) for r in values} for phase,values in records.items()}
    expected_keys={(r['id'],i) for r in rows for i in range(samples)}
    for phase in records:
        if len(done[phase])!=len(records[phase]) or not done[phase]<=expected_keys: raise ValueError('Invalid prediction journal')
    tasks={}
    buffers={}
    total=len(rows)*samples*2
    for row in rows:
        prompt=prompts[row['id']]
        for sample_index in range(samples):
            seed=int.from_bytes(hashlib.sha256(f"42:{row['id']}:{sample_index}".encode()).digest()[:4],'big')
            for phase in outputs:
                if (row['id'],sample_index) in done[phase]: continue
                request_id=f'{phase}:{row["id"]}:{sample_index}'
                params=SamplingParams(n=1,temperature=1.0,top_p=0.95,top_k=-1,min_p=0.0,
                    presence_penalty=0.0,max_tokens=max_tokens,seed=seed,stop_token_ids=eos,
                    output_kind=RequestOutputKind.DELTA)
                tasks[request_id]=(phase,row,sample_index,seed)
                buffers[request_id]={'parts':[],'tokens':0,'started':None}
                engine.add_request(request_id,{'prompt_token_ids':prompt},params,
                                   lora_request=lora if phase=='trained_vllm' else None)
    started=time.monotonic()
    last=0
    compat_results=[]
    while engine.has_unfinished_requests():
        for result in engine.step():
            request_id=result.request_id
            if request_id not in tasks: continue
            phase,row,sample_index,seed=tasks[request_id]
            buffer=buffers[request_id]
            if buffer['started'] is None: buffer['started']=time.monotonic()
            if result.outputs:
                item=result.outputs[0]
                buffer['parts'].append(item.text)
                buffer['tokens']+=len(item.token_ids)
            if not result.finished: continue
            text=''.join(buffer['parts'])
            boxed=last_boxed_answer(text)
            parsed=[]
            correct=False
            fallback=False
            try:
                if boxed is not None:
                    parsed=parse(math_answer_markup(boxed),fallback_mode='no_fallback')
                    correct=bool(parsed and verify(golds[row['id']],parsed,timeout_seconds=5))
            except Exception:
                normalize=lambda x:x.replace('$','').replace(' ','').lower().strip()
                correct=bool(boxed is not None and normalize(boxed)==normalize(str(row['answer'])))
                fallback=True
            record={'id':row['id'],'sample_index':sample_index,'sample_seed':seed,
                'prediction':text,'boxed_answer':boxed,'answer':row['answer'],'correct':correct,
                'extraction_failed':not bool(parsed),'grader_exception_fallback':fallback,
                'tokens':buffer['tokens'],'truncated':result.outputs[0].finish_reason=='length',
                'seconds_active':time.monotonic()-buffer['started'],'backend':'vllm'}
            records[phase].append(record)
            done[phase].add((row['id'],sample_index))
            if args.compat: compat_results.append(record)
            else:
                with (outputs[phase]/'predictions.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            del buffers[request_id]
            print(f'COMPLETE {phase} {len(records[phase])}/{len(rows)*samples} tokens={record["tokens"]}',flush=True)
        now=time.monotonic()
        if now-last>=30 and not args.compat:
            atomic_json(root/'vllm_state.json',{'status':'evaluating','completed':{k:len(v) for k,v in records.items()},
                'total':total,'seconds_this_invocation':now-started,
                'active_generated_tokens':sum(v['tokens'] for v in buffers.values()),
                'completed_generated_tokens':sum(r['tokens'] for v in records.values() for r in v),
                'updated_unix':time.time()})
            last=now
    if args.compat:
        if len(compat_results)!=2: raise RuntimeError('Mixed base/LoRA compatibility check failed')
        atomic_json(root/'vllm_compat.json',{'status':'passed','version':vllm.__version__,'outputs':compat_results})
    else:
        summaries={}
        metric='accuracy' if samples==1 else f'Avg@{samples}'
        for phase,values in records.items():
            expected_count=len(rows)*samples
            if len(values)!=expected_count: raise ValueError(f'Expected {expected_count} completions per policy')
            summary={'n':len(rows),'samples_per_prompt':samples,'evaluated_completions':expected_count,
                'dataset':args.dataset_label,'metric':metric,
                'accuracy':sum(r['correct'] for r in values)/expected_count,
                'truncation_rate':sum(r['truncated'] for r in values)/expected_count,
                'extraction_failure_rate':sum(r['extraction_failed'] for r in values)/expected_count,
                'generated_tokens':sum(r['tokens'] for r in values),'backend':'vllm',
                'thinking':True,'max_new_tokens':max_tokens,'temperature':1.0}
            atomic_json(outputs[phase]/'summary.json',summary)
            summaries[phase]=summary
        before={(r['id'],r['sample_index']):r for r in records['baseline_vllm']}
        after={(r['id'],r['sample_index']):r for r in records['trained_vllm']}
        if before.keys()!=after.keys(): raise ValueError('Unpaired evaluation keys')
        per_question=defaultdict(lambda:{'before_correct':0,'after_correct':0})
        for key in before:
            if before[key]['sample_seed']!=after[key]['sample_seed']: raise ValueError('Seed mismatch')
            per_question[key[0]]['before_correct']+=int(before[key]['correct'])
            per_question[key[0]]['after_correct']+=int(after[key]['correct'])
        comparison={'training_steps':100,
            'training_batch_size':config['train']['micro_batch_size']*config['train']['gradient_accumulation_steps'],
            'training_micro_batch_size':config['train']['micro_batch_size'],
            'training_gradient_accumulation_steps':config['train']['gradient_accumulation_steps'],
            'questions':len(rows),'samples_per_question':samples,
            'metric':f'{metric} on {args.dataset_label}','max_new_tokens':max_tokens,'backend':'vllm',
            'baseline':summaries['baseline_vllm'],'trained':summaries['trained_vllm'],
            'delta_percentage_points':100*(summaries['trained_vllm']['accuracy']-summaries['baseline_vllm']['accuracy']),
            'improved_completions':sum(not before[k]['correct'] and after[k]['correct'] for k in before),
            'regressed_completions':sum(before[k]['correct'] and not after[k]['correct'] for k in before),
            'per_question':dict(per_question),'identity':identity}
        atomic_json(root/'comparison.json',comparison)
        atomic_json(root/'vllm_state.json',{'status':'complete','completed':total,'updated_unix':time.time()})
        print(json.dumps(comparison,indent=2),flush=True)
    # Explicit engine shutdown also releases its spawned worker processes.
    client=getattr(engine,'engine_core',None)
    if client is not None and hasattr(client,'shutdown'): client.shutdown()


if __name__=='__main__': main()

"""Run resumable AIME24 then AIME25 comparisons sequentially on one GPU."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,help='Completed 100-step training directory')
    parser.add_argument('--output-root')
    parser.add_argument('--data-root',default='data/processed/aime')
    parser.add_argument('--max-num-seqs',type=int,default=8)
    args=parser.parse_args()
    training=Path(args.root).resolve()
    config=yaml.safe_load((training/'run_config.yaml').read_text())
    microbatch=config['train']['micro_batch_size']
    accumulation=config['train']['gradient_accumulation_steps']
    root=Path(args.output_root).resolve() if args.output_root else training/'aime24_aime25_38912_s12'
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'suite.lock').open('a')
    try:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('An AIME comparison controller already owns this run') from error
    final=training/'training/checkpoints/step_000100'
    if not (final/'training_state.pt').is_file():
        raise ValueError('Step-100 checkpoint is incomplete')
    data_root=Path(args.data_root).resolve()
    manifest=json.loads((data_root/'manifest.json').read_text())
    datasets={}
    for name in ['aime24','aime25']:
        source=manifest['datasets'][name]
        path=data_root/f'{name}.jsonl'
        if source['count']!=30 or digest(path)!=source['processed_sha256']:
            raise ValueError(f'Prepared dataset identity differs: {name}')
        datasets[name]={**source,'data_file':str(path)}
    plan={'training_root':str(training),'checkpoint':str(final),
        'student_adapter_sha256':digest(final/'adapters/student/adapter_model.safetensors'),
        'datasets':datasets,'parameters':{'thinking':True,'samples_per_prompt':12,
        'max_new_tokens':38912,'temperature':1.0,'top_p':0.95,'top_k':-1,'min_p':0.0,
        'presence_penalty':0.0,'max_model_len':40960},'metric':'Avg@12',
        'completions_per_policy_per_dataset':360,'total_completions':1440,
        'official_paper':'https://arxiv.org/html/2601.18734v3#A2',
        'execution':{'tensor_parallel_size':1,'max_num_seqs':args.max_num_seqs,'backend':'vllm==0.11.0'}}
    plan_path=root/'evaluation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Existing AIME suite belongs to different weights, data or parameters')
    atomic_json(plan_path,plan)
    progress={name:{'status':'queued'} for name in datasets}
    summaries={}
    for name,source in datasets.items():
        output=root/name
        output.mkdir(exist_ok=True)
        command=[sys.executable,'-u','scripts/evaluate_vllm_pair.py','--root',str(training),
            '--output-root',str(output),'--data-file',source['data_file'],'--dataset-label',name.upper(),
            '--limit','0','--samples-per-prompt','12','--max-new-tokens','38912',
            '--max-model-len','40960','--max-num-seqs',str(args.max_num_seqs)]
        atomic_json(output/'launch_command.json',command)
        for attempt in range(1,4):
            if (output/'comparison.json').exists():
                summary=json.loads((output/'comparison.json').read_text())
                identity=summary['identity']
                if (identity['data_sha256']!=source['processed_sha256'] or
                    identity['student_adapter_sha256']!=plan['student_adapter_sha256'] or
                    summary['questions']!=30 or summary['samples_per_question']!=12 or
                    summary['max_new_tokens']!=38912):
                    raise ValueError(f'Unexpected completed comparison: {name}')
                summaries[name]=summary
                progress[name]={'status':'complete','completed':720}
                break
            with (output/'vllm_pair.log').open('a') as log:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (output/'vllm_pair.pid').write_text(str(process.pid))
                print(f'START {name} attempt={attempt} pid={process.pid}',flush=True)
                while process.poll() is None:
                    child_state=output/'vllm_state.json'
                    state=json.loads(child_state.read_text()) if child_state.exists() else {'status':'starting'}
                    progress[name]={**state,'pid':process.pid,'attempt':attempt}
                    atomic_json(root/'suite_state.json',{'status':'running','current_dataset':name,
                        'datasets':progress,'total_completions':1440,'updated_unix':time.time()})
                    time.sleep(10)
            print(f'EXIT {name} code={process.returncode}',flush=True)
            if process.returncode!=0:
                if attempt==3:
                    atomic_json(root/'suite_state.json',{'status':'failed','dataset':name,
                        'datasets':progress,'updated_unix':time.time()})
                    raise RuntimeError(f'{name} evaluation failed three times; inspect its log')
                continue
            # Read and validate the completed output after a successful child exit.
            summary=json.loads((output/'comparison.json').read_text())
            summaries[name]=summary
            progress[name]={'status':'complete','completed':720}
            break
    result={'metric':'Avg@12','training_steps':100,
        'training_batch_size':microbatch*accumulation,
        'training_micro_batch_size':microbatch,'training_gradient_accumulation_steps':accumulation,
        'parameters':plan['parameters'],'datasets':summaries,
        'baseline_macro_average':sum(v['baseline']['accuracy'] for v in summaries.values())/2,
        'trained_macro_average':sum(v['trained']['accuracy'] for v in summaries.values())/2}
    result['macro_delta_percentage_points']=100*(result['trained_macro_average']-result['baseline_macro_average'])
    atomic_json(root/'comparison.json',result)
    atomic_json(root/'suite_state.json',{'status':'complete','datasets':progress,
        'total_completions':1440,'completed':1440,'updated_unix':time.time()})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

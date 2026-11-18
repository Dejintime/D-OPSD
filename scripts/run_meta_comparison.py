"""Supervise a launched training job and resumable paired official evaluations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    config=yaml.safe_load((root/'run_config.yaml').read_text())
    config['model']['offload_inner_activations']=False
    config['model']['offload_teacher_activations']=False
    config['evaluation']={'max_new_tokens':38912,'thinking':True,'temperature':1.0,
                          'top_p':0.95,'top_k':0,'samples_per_prompt':12,
                          'prompt_style':'opsd_eval','answer_extraction':'last_boxed'}
    cfg=root/'evaluation_config_official.yaml'
    cfg.write_text(yaml.safe_dump(config,allow_unicode=True))
    train_pid=int((root/'training.pid').read_text())
    final=root/'training/checkpoints/step_000100'
    stages={'baseline_official':None,'trained_official':None}
    attempts=defaultdict(int)
    logs={}
    def write_state(status):
        temporary=root/'comparison_state.tmp'
        temporary.write_text(json.dumps({'status':status,'training_checkpoint_complete':(final/'training_state.pt').exists(),
            'attempts':dict(attempts),'eval_pids':{name:p.pid for name,p in stages.items() if p and p.poll() is None},
            'updated_unix':time.time()}))
        temporary.replace(root/'comparison_state.json')
    def train_alive():
        try:
            c=Path(f'/proc/{train_pid}/cmdline').read_bytes()
            return b'scripts/train.py' in c and b'meta_qwen3_100_run.yaml' in c
        except FileNotFoundError: return False
    def can_launch():
        known={train_pid}|{p.pid for p in stages.values() if p is not None}
        command=['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']
        result=subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
        other_mib=0
        for line in result.stdout.splitlines():
            pid,memory=line.split(',')
            if int(pid) not in known: other_mib+=int(memory.strip())
        running=sum(p is not None and p.poll() is None for p in stages.values())
        # Reserve full long-context cache budgets rather than admitting jobs based
        # on their small early-generation memory footprint. No other jobs are killed.
        reserved=other_mib+running*10*1024+(12*1024 if train_alive() else 0)
        return reserved+10*1024<=22*1024
    while True:
        trained=(final/'training_state.pt').exists()
        if not trained and not train_alive():
            write_state('training_failed_or_interrupted')
            raise RuntimeError('Training stopped before step 100; inspect training.log')
        for name,process in list(stages.items()):
            if process is not None and process.poll() is not None:
                logs.pop(name).close()
                stages[name]=None
                print(f'{name} exit={process.returncode}',flush=True)
        for name in stages:
            if (root/name/'summary.json').exists() or stages[name] is not None: continue
            if name=='trained_official' and not trained: continue
            if attempts[name]>=3:
                write_state('evaluation_failed')
                raise RuntimeError(f'{name} failed three times; inspect its log')
            if not can_launch(): continue
            command=[sys.executable,'-u','scripts/evaluate.py','--config',str(cfg),
                     '--split','dev','--limit','50','--sample-seed','42','--output',str(root/name),'--resume']
            command+=['--initial'] if name=='baseline_official' else ['--checkpoint',str(final)]
            logs[name]=(root/f'{name}.log').open('a')
            stages[name]=subprocess.Popen(command,stdout=logs[name],stderr=subprocess.STDOUT)
            attempts[name]+=1
            (root/f'{name}.pid').write_text(str(stages[name].pid))
            print(f'START {name} pid={stages[name].pid} attempt={attempts[name]}',flush=True)
        if trained and all((root/name/'summary.json').exists() for name in stages): break
        write_state('training_and_official_evaluation')
        time.sleep(10)
    predictions={}
    for name in stages:
        records=[json.loads(line) for line in (root/name/'predictions.jsonl').read_text().splitlines()]
        predictions[name]={(r['id'],r['sample_index']):r for r in records}
        if len(predictions[name])!=600: raise ValueError('Expected 50 questions times 12 completions')
    before,after=predictions['baseline_official'],predictions['trained_official']
    if before.keys()!=after.keys(): raise ValueError('Unpaired evaluations')
    by_question=defaultdict(lambda:{'before_correct':0,'after_correct':0})
    for key in before:
        if before[key]['sample_seed']!=after[key]['sample_seed']: raise ValueError('Unpaired sample seeds')
        by_question[key[0]]['before_correct']+=int(before[key]['correct'])
        by_question[key[0]]['after_correct']+=int(after[key]['correct'])
    b=json.loads((root/'baseline_official/summary.json').read_text())
    a=json.loads((root/'trained_official/summary.json').read_text())
    comparison={'training_steps':100,'training_batch_size':1,'questions':50,'samples_per_question':12,
                'thinking':True,'max_new_tokens':38912,'temperature':1.0,
                'metric':'Avg@12 on held-out OpenThoughts developer questions, not an official benchmark leaderboard',
                'baseline':b,'trained':a,'delta_percentage_points':100*(a['accuracy']-b['accuracy']),
                'improved_completions':sum(not before[k]['correct'] and after[k]['correct'] for k in before),
                'regressed_completions':sum(before[k]['correct'] and not after[k]['correct'] for k in before),
                'per_question':dict(by_question)}
    (root/'comparison_official.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False))
    write_state('complete')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__': main()

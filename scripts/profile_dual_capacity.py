"""Isolated dual-LoRA memory probes; never changes the production Trainer.

Synthetic continuations stress the longest eligible prompt shapes. Accumulation
forms ONE shared inner SGD update, then accumulates its exact teacher
hypergradient and the updated-teacher pilot student gradient in separate passes.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def save(path,value):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


def _chunk_hypergradient(bundle,views,direction,inner_lr,temperature=1.0,
                        token_clip=None,temperature_squared=True):
    import torch
    primal,tangent=bundle.student_jvp(views[0],direction)
    dtype=torch.float64 if primal.dtype==torch.float64 else torch.float32
    log_s=torch.log_softmax(primal.to(dtype)/temperature,-1).detach()
    tangent=tangent.to(dtype).detach()
    logits=bundle.forward(views[1],'teacher')
    log_t=torch.log_softmax(logits.to(dtype)/temperature,-1)
    probability=log_t.exp()
    with torch.no_grad():
        mask=torch.ones_like(probability) if token_clip is None else (
            probability.detach()*(log_t.detach()-log_s)<=token_clip).to(dtype)
        coefficient=((log_s.exp()*tangent).sum(-1,keepdim=True)-tangent)*mask
    per_token=(probability*coefficient).sum(-1)
    valid=views[0]['response_mask']
    objective=per_token.masked_select(valid).mean()
    scale=(temperature**2 if temperature_squared else 1.0)/temperature
    return torch.autograd.grad(-inner_lr*scale*objective,tuple(bundle.parameters('teacher').values()))


def accumulated_inner_meta_grad(bundle,chunks,quiz,inner_lr,callback=None,**options):
    """Exact one-step SGD hypergradient, without retaining graphs across chunks."""
    import torch
    from dual_opsd.losses import forward_kl
    student=bundle.parameters('student')
    counts=[int(views[0]['response_mask'].sum()) for views in chunks]
    weights=[n/sum(counts) for n in counts]
    accumulated=None
    inner_value=0.0
    for index,(views,weight) in enumerate(zip(chunks,weights)):
        with torch.no_grad():target=bundle.forward(views[1],'teacher').detach()
        logits=bundle.forward(views[0],'student')
        loss=forward_kl(target,logits,mask=views[0]['response_mask'],**options)
        inner_value+=loss.item()*weight
        gradients=torch.autograd.grad(loss,tuple(student.values()))
        if accumulated is None:accumulated=[g.detach()*weight for g in gradients]
        else:
            for total,g in zip(accumulated,gradients):total.add_(g.detach(),alpha=weight)
        del target,logits,loss,gradients
        if callback:callback('inner',index+1)
    fast={name:(p-inner_lr*g).detach().requires_grad_(True)
          for (name,p),g in zip(student.items(),accumulated)}
    del accumulated
    with torch.no_grad():reference=bundle.forward(quiz[1],'reference').detach()
    quiz_options={k:v for k,v in options.items() if k!='token_clip'}
    after=forward_kl(reference,bundle.forward(quiz[0],'student',fast),
        mask=quiz[0]['response_mask'],**quiz_options)
    quiz_value=after.item()
    values=torch.autograd.grad(after,tuple(fast.values()))
    direction=dict(zip(student,values))
    del fast,reference,after,values
    accumulated=None
    for index,(views,weight) in enumerate(zip(chunks,weights)):
        gradients=_chunk_hypergradient(bundle,views,direction,inner_lr,**options)
        if accumulated is None:accumulated=[g.detach()*weight for g in gradients]
        else:
            for total,g in zip(accumulated,gradients):total.add_(g.detach(),alpha=weight)
        del gradients
        if callback:callback('meta',index+1)
    return accumulated,{'inner_loss':inner_value,'quiz_after':quiz_value}


def prepare(config,root):
    from transformers import AutoTokenizer
    from dual_opsd.prompts import student_prompt,teacher_prompt
    tokenizer=AutoTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    cfg=config['rollout'];prompts=config['prompts'];result={}
    for split in ['distill','quiz']:
        selected={'student':[],'teacher':[]};kept=0
        with (Path(config['data']['root'])/f'{split}.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                student=student_prompt(tokenizer,row['problem'],prompts['style'],prompts['student_thinking'])
                teacher=teacher_prompt(tokenizer,row,config['data']['privilege'],prompts['style'],prompts['teacher_thinking'])
                if max(len(student),len(teacher))+cfg['max_new_tokens']>cfg['max_total_tokens']:continue
                if len(student)+cfg['max_new_tokens']>cfg['max_student_tokens']:continue
                kept+=1
                for name,value in [('student',student),('teacher',teacher)]:
                    if len(value)>len(selected[name]):selected[name]=value
        if not kept:raise ValueError(f'No eligible {split} rows')
        result[split]={**selected,'kept':kept}
    token=tokenizer.encode(' Therefore, the answer is 42.',add_special_tokens=False)
    result['response']=(token*(cfg['max_new_tokens']//len(token)+1))[:cfg['max_new_tokens']]
    result['pad_token_id']=tokenizer.pad_token_id
    save(root/'shapes.json',result)
    save(root/'shape_summary.json',{split:{'eligible':result[split]['kept'],
        'student_total':len(result[split]['student'])+len(result['response']),
        'teacher_total':len(result[split]['teacher'])+len(result['response'])} for split in ['distill','quiz']})


def worker(args,config):
    import psutil
    import torch
    from dual_opsd.data import make_batch_view
    from dual_opsd.losses import forward_kl,loss_options
    from dual_opsd.meta import teacher_meta_grad
    from dual_opsd.models import build_model
    from dual_opsd.utils import seed_all,norm
    root=Path(args.output);result_path=root/f'b{args.micro}_a{args.accum}.json'
    process=psutil.Process();stop=threading.Event()
    stats={'rss_peak_gib':process.memory_info().rss/2**30,
        'host_available_min_gib':psutil.virtual_memory().available/2**30}
    def monitor():
        while not stop.wait(0.05):
            stats['rss_peak_gib']=max(stats['rss_peak_gib'],process.memory_info().rss/2**30)
            stats['host_available_min_gib']=min(stats['host_available_min_gib'],psutil.virtual_memory().available/2**30)
            if stats['rss_peak_gib']>96 or stats['host_available_min_gib']<20:
                save(result_path,{'status':'host_memory_guard','micro_batch_size':args.micro,
                    'accumulation':args.accum,**stats})
                os._exit(42)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    started=time.monotonic();row={'micro_batch_size':args.micro,'accumulation':args.accum,
        'quiz_batch_size':config['train']['quiz_batch_size'],
        'unequal_prompt_response_lengths':args.unequal_shapes,
        'effective_student_examples':args.micro*args.accum,'algorithm':'one_shared_inner_sgd_step_microbatch_accumulation',
        'continuations':'synthetic fixed-length; no generation or accuracy measurement','episode_memory':[]}
    try:
        if psutil.virtual_memory().available<24*2**30:raise RuntimeError('Insufficient free host RAM for safe offload probe')
        seed_all(42)
        bundle,_=build_model(config);bundle.configure_checkpointing(False)
        student=list(bundle.parameters('student').values());teacher=list(bundle.parameters('teacher').values())
        train=config['train']
        optimizers=[torch.optim.AdamW(student,lr=train['student_lr'],weight_decay=0),
            torch.optim.AdamW(teacher,lr=train['teacher_lr'],weight_decay=0)]
        # Allocate BOTH AdamW moment sets before measuring steady-state meta work.
        for opt,params in zip(optimizers,[student,teacher]):
            for p in params:p.grad=torch.zeros_like(p)
            opt.step();opt.zero_grad(set_to_none=True)
        shapes=json.loads((root/'shapes.json').read_text())
        def views(split):
            batch=args.micro if split=='distill' else train['quiz_batch_size']
            return tuple(make_batch_view([
                shapes[split][role][:-1] if args.unequal_shapes and index%2 else shapes[split][role]
                for index in range(batch)], [
                shapes['response'][:-1] if args.unequal_shapes and index%2 else shapes['response']
                for index in range(batch)],bundle.device,shapes['pad_token_id'])
                for role in ['student','teacher'])
        distill,quiz=views('distill'),views('quiz')
        options=loss_options(config['loss'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        row['rss_before_compute_gib']=process.memory_info().rss/2**30
        compute_start=time.monotonic()
        def record(phase,index):
            torch.cuda.synchronize()
            row['episode_memory'].append({'phase':phase,'episode':index,
                'rss_gib':process.memory_info().rss/2**30,'allocated_gib':torch.cuda.memory_allocated()/2**30})
            if index in {1,4,8,16,32}:
                print(f'{phase.upper()} {index}/{args.accum} '+json.dumps(row['episode_memory'][-1]),flush=True)
        gradients,metrics=accumulated_inner_meta_grad(bundle,[distill]*args.accum,quiz,
            train['inner_lr'],callback=record,**options)
        if not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Invalid meta-gradient')
        for p,g in zip(teacher,gradients):p.grad=g.detach()
        del gradients
        row['teacher_gradient_norm']=norm(p.grad for p in teacher)
        torch.nn.utils.clip_grad_norm_(teacher,train['max_grad_norm'])
        optimizers[1].step();optimizers[1].zero_grad(set_to_none=True)
        for index in range(args.accum):
            with torch.no_grad():target=bundle.forward(distill[1],'teacher').detach()
            loss=forward_kl(target,bundle.forward(distill[0],'student'),
                mask=distill[0]['response_mask'],**options)
            if not torch.isfinite(loss):raise FloatingPointError('Invalid student loss')
            (loss/args.accum).backward()
            del target,loss
            torch.cuda.synchronize()
            row['episode_memory'].append({'phase':'student','episode':index+1,
                'rss_gib':process.memory_info().rss/2**30,'allocated_gib':torch.cuda.memory_allocated()/2**30})
            if (index+1) in {1,4,8,16,32}:
                print(f'STUDENT {index+1}/{args.accum} '+json.dumps(row['episode_memory'][-1]),flush=True)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in student):
            raise FloatingPointError('Invalid student gradients')
        row['student_gradient_norm']=norm(p.grad for p in student)
        torch.nn.utils.clip_grad_norm_(student,train['max_grad_norm'])
        optimizers[0].step();optimizers[0].zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row.update(status='ok',compute_seconds=time.monotonic()-compute_start)
    except torch.OutOfMemoryError as error:
        row.update(status='oom',error=str(error).split('\n')[0])
    except Exception:
        row.update(status='error',error=traceback.format_exc())
    finally:
        stop.set();thread.join()
        if torch.cuda.is_initialized():
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        row.update(stats,seconds_total=time.monotonic()-started)
        save(result_path,row);print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/meta_qwen3.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--micro',type=int,default=1)
    parser.add_argument('--accum',type=int,default=1)
    parser.add_argument('--fixed-micro',type=int,default=0)
    parser.add_argument('--accumulations',nargs='+',type=int,default=[8,16])
    parser.add_argument('--unequal-shapes',action='store_true')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ.setdefault('OMP_NUM_THREADS','1')
    import yaml
    config=yaml.safe_load(Path(args.config).read_text())
    if args.worker:return worker(args,config)
    root=Path(args.output)
    if root.exists():raise ValueError('Use a new output directory')
    root.mkdir(parents=True)
    save(root/'config.json',config)
    save(root/'progress.json',{'status':'preparing_worst_shapes'})
    prepare(config,root)
    rows=[]
    def run(micro,accum):
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--config',args.config,
            '--output',str(root),'--gpu',args.gpu,'--micro',str(micro),'--accum',str(accum)]
        if args.unequal_shapes:command.append('--unequal-shapes')
        with (root/f'b{micro}_a{accum}.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            save(root/'progress.json',{'status':'running','micro':micro,'accum':accum,'pid':child.pid})
            while child.poll() is None:time.sleep(5)
        path=root/f'b{micro}_a{accum}.json'
        if not path.exists():raise RuntimeError(f'Probe exited without result: {micro}, {accum}; see log')
        row=json.loads(path.read_text());rows.append(row);save(root/'results.json',rows)
        print('RESULT',json.dumps({k:v for k,v in row.items() if k!='episode_memory'}),flush=True)
        if row['status'] not in {'ok','oom'}:raise RuntimeError(row.get('error',row['status']))
        return row
    if args.fixed_micro:
        for accumulation in args.accumulations:
            if accumulation<1:raise ValueError('Accumulation must be positive')
            row=run(args.fixed_micro,accumulation)
            if row['status']!='ok':break
        save(root/'summary.json',{'micro_batch_size':args.fixed_micro,
            'quiz_batch_size':config['train']['quiz_batch_size'],
            'all_probes_passed':all(r['status']=='ok' for r in rows),
            'production_trainer_supports_only_micro1_accum1':False,
            'accumulation_semantics':'one shared inner SGD step over all microbatches, then exact accumulated teacher hypergradient and pilot student update',
            'synthetic_worst_shapes':True,
            'rows':[{k:v for k,v in r.items() if k!='episode_memory'} for r in rows]})
        save(root/'progress.json',{'status':'complete'})
        return
    low=0;high=None
    for batch in [1,2,4,8,16]:
        row=run(batch,1)
        if row['status']=='oom':high=batch;break
        low=batch
    if low==0:raise RuntimeError('Batch 1 failed; inspect resource limits')
    if high is not None:
        for batch in range(low+1,high):
            row=run(batch,1)
            if row['status']=='oom':break
            low=batch
    safe=[r['micro_batch_size'] for r in rows if r['status']=='ok' and r['peak_reserved_gib']<=22]
    recommended=max(safe) if safe else 1
    # A power-of-two microbatch gives convenient effective batch-32 settings.
    practical=max(b for b in [1,2,4,8,16] if b<=recommended and any(
        r['status']=='ok' and r['micro_batch_size']==b for r in rows))
    for accumulation in sorted({4,32//practical}):
        if accumulation>1:run(practical,accumulation)
    save(root/'summary.json',{'max_observed_micro':low,'first_oom_micro':high,
        'recommended_micro_with_2gib_headroom':practical,'tested_accumulation':32//practical,
        'effective_student_examples':32,'production_trainer_supports_only_micro1_accum1':False,
        'accumulation_semantics':'one shared inner SGD step over all microbatches, then exact accumulated teacher hypergradient and pilot student update',
        'synthetic_worst_shapes':True,'rows':[{k:v for k,v in r.items() if k!='episode_memory'} for r in rows]})
    save(root/'progress.json',{'status':'complete'})


if __name__=='__main__':main()

"""Bounded real-model batch calibration, followed by one effective-batch update."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from dual_opsd.data import make_batch_view
from dual_opsd.losses import forward_kl, loss_options
from dual_opsd.models import build_model
from dual_opsd.trainer import Trainer
from dual_opsd.utils import load_config, norm, save_json, seed_all


def calibrate(trainer, batches):
    bundle = trainer.bundle
    records = trainer.prepared['distill']
    # Conservative maxima over the actual eligible data, not only short samples.
    student = max(records, key=lambda x: len(x[1]))[1]
    teacher = max(records, key=lambda x: len(x[2]))[2]
    length = trainer.config['rollout']['max_new_tokens']
    token = trainer.tokenizer.encode(' Therefore, the answer is 42.', add_special_tokens=False)
    response = (token * (length//len(token)+1))[:length]
    initial = {n: p.detach().cpu().clone() for n,p in trainer.student.items()}
    free, total = torch.cuda.mem_get_info()
    available = free + torch.cuda.memory_allocated()
    results = []

    def trial(batch):
        views = [make_batch_view([prompt]*batch, [response]*batch, bundle.device,
                                trainer.tokenizer.pad_token_id) for prompt in (student, teacher)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            target = bundle.forward(views[1], 'reference').detach()
        loss = forward_kl(target, bundle.forward(views[0], 'student'),
                          mask=views[0]['response_mask'], **loss_options(trainer.config['loss']))
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainer.student.values()):
            raise FloatingPointError('Invalid batch-probe gradient')
        torch.nn.utils.clip_grad_norm_(tuple(trainer.student.values()), trainer.config['train']['max_grad_norm'])
        trainer.student_opt.step()  # Include AdamW moment allocation in the peak.
        torch.cuda.synchronize()
        return {'micro_batch_size':batch, 'status':'ok', 'seconds_excluding_rollout':time.perf_counter()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'teacher_total_tokens':len(teacher)+length,'student_total_tokens':len(student)+length,
                'response_tokens_per_example':length,'loss':loss.item()}

    for batch in batches:
        trainer.student_opt.zero_grad(set_to_none=True)
        trainer.student_opt.state.clear()
        with torch.no_grad():
            for name,p in trainer.student.items(): p.copy_(initial[name])
        gc.collect()
        torch.cuda.empty_cache()
        try:
            row = trial(batch)
        except torch.OutOfMemoryError as error:
            row = {'micro_batch_size':batch, 'status':'oom', 'error':str(error).split('\n')[0]}
        results.append(row)
        save_json(trainer.output/'batch_scan.json', results)
        print(json.dumps(row), flush=True)
        trainer.student_opt.zero_grad(set_to_none=True)
        trainer.student_opt.state.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if row['status'] == 'oom': break
    with torch.no_grad():
        for name,p in trainer.student.items(): p.copy_(initial[name])
    candidates = [r['micro_batch_size'] for r in results if r['status']=='ok' and
                  r['peak_reserved_gib']*2**30 < available-2*1024**3 and 32%r['micro_batch_size']==0]
    if not candidates:
        raise RuntimeError('No tested batch retained 2 GiB headroom under worst eligible shapes')
    # Sampling is sequential in this implementation; larger B mainly changes
    # memory use, not autoregressive generation parallelism.
    return max(candidates), results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/opsd_qwen3.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', nargs='+',type=int,default=[1,2,4,8])
    args=parser.parse_args()
    if Path(args.output).exists(): raise ValueError('Use a new output directory')
    if any(b<1 or 32%b for b in args.batches): parser.error('Batches must divide 32')
    config=load_config(args.config)
    if config['experiment']['method']!='opsd': raise ValueError('This scan is for the fixed-teacher baseline')
    seed_all(config['experiment']['seed'])
    bundle,tokenizer=build_model(config)
    trainer=Trainer(config,bundle,tokenizer,args.output)
    chosen,scan=calibrate(trainer,args.batches)
    trainer.microbatch=chosen
    trainer.accumulation=32//chosen
    trainer.config['train']['micro_batch_size']=chosen
    trainer.config['train']['gradient_accumulation_steps']=trainer.accumulation
    (trainer.output/'config.yaml').write_text(yaml.safe_dump(trainer.config,allow_unicode=True))
    # Probe gradients/weights have been discarded. This is a fresh real on-policy
    # update, with 32 source examples and a checkpoint saved afterward.
    seed_all(config['experiment']['seed'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f'START_REAL_UPDATE microbatch={chosen} accumulation={trainer.accumulation}',flush=True)
    metrics=trainer.train_step()
    trainer.save()
    save_json(trainer.output/'real_update.json',metrics)
    save_json(trainer.output/'recommendation.json',{
        'micro_batch_size':chosen,'gradient_accumulation_steps':trainer.accumulation,'effective_batch_size':32,
        'safety_headroom_gib':2,'batch_scan_kind':'worst_actual_prompt_shapes_with_synthetic_1024_token_continuation',
        'completed_real_optimizer_steps':1,'sampling':'sequential per example, batched scoring/backward',
        'real_update':metrics})
    print(json.dumps(metrics),flush=True)


if __name__=='__main__': main()

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .data import file_hash, make_batch_view, read_jsonl
from .losses import forward_kl, loss_options
from .meta import teacher_meta_grad, accumulated_teacher_meta_grad
from .prompts import student_prompt, teacher_prompt
from .rollout import sample_rollout
from .utils import environment, load_checkpoint, norm, save_checkpoint, save_json


class Trainer:
    def __init__(self, config, bundle, tokenizer, output_dir):
        self.config, self.bundle, self.tokenizer = config, bundle, tokenizer
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        train = config['train']
        self.method = config['experiment']['method']
        if self.method not in {'opsd', 'meta_opsd', 'sft'}:
            raise ValueError('Unknown method')
        self.microbatch = train.get('micro_batch_size', 1)
        self.accumulation = train.get('gradient_accumulation_steps', 1)
        if self.microbatch < 1 or self.accumulation < 1:
            raise ValueError('Batch sizes must be positive')
        if train.get('inner_steps', 1) != 1:
            raise ValueError('Only one differentiable inner step is implemented')
        if self.method == 'meta_opsd':
            if train.get('quiz_batch_size',1)<1:
                raise ValueError('Quiz batch size must be positive')
            if self.microbatch*self.accumulation>1 and bundle.meta_gradient!='jvp':
                raise ValueError('Batched meta training requires the validated shared-inner JVP path')
            if train.get('gradient_checkpointing', False):
                raise ValueError('Shared-adapter meta checkpoint recomputation has not been validated')
        if config['loss']['kind'] not in {'forward_kl', 'clipped_forward_kl'}:
            raise ValueError('Unsupported divergence')
        if config['loss']['kind'] == 'clipped_forward_kl' and not config['loss'].get('token_clip'):
            raise ValueError('Clipped loss requires a positive token_clip')
        bundle.configure_checkpointing(train.get('gradient_checkpointing', False))
        self.student = bundle.parameters('student')
        self.teacher = bundle.parameters('teacher')
        assert self.student and self.teacher
        assert not ({id(p) for p in self.student.values()} & {id(p) for p in self.teacher.values()})
        if train.get('student_optimizer', 'sgd') == 'adamw':
            self.student_opt = torch.optim.AdamW(self.student.values(), lr=train['student_lr'],
                betas=tuple(train.get('adam_betas', [0.9, 0.999])), eps=train.get('adam_epsilon', 1e-8),
                weight_decay=train.get('student_weight_decay', 0.0))
        elif train.get('student_optimizer', 'sgd') == 'sgd':
            self.student_opt = torch.optim.SGD(self.student.values(), lr=train['student_lr'])
        else:
            raise ValueError('Unknown student optimizer')
        self.teacher_opt = torch.optim.AdamW(self.teacher.values(), lr=train.get('teacher_lr', 1e-5),
                                            weight_decay=train.get('teacher_weight_decay', 0.0))
        linear = train.get('lr_scheduler', 'constant') == 'linear'
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.student_opt,
            lambda step: max(0.0, 1 - step / train['max_steps']) if linear else 1.0)
        root = Path(config['data']['root'])
        self.manifest = json.loads((root / 'manifest.json').read_text())
        for split, metadata in self.manifest.get('splits', {}).items():
            if file_hash(root / f'{split}.jsonl') != metadata['sha256']:
                raise ValueError(f'Dataset hash mismatch: {split}')
        self.prepared, self.skipped = {}, {}
        splits = ('distill', 'quiz') if self.method == 'meta_opsd' else ('distill',)
        for split in splits:
            examples = read_jsonl(root / f'{split}.jsonl')
            prepared = [self.prepare(e) for e in examples]
            self.prepared[split] = [e for e in prepared if e is not None]
            self.skipped[split] = len(examples) - len(self.prepared[split])
            if not self.prepared[split]:
                raise ValueError(f'No {split} examples fit token budget')
        self.sampler = {key: {'order': [], 'position': 0} for key in self.prepared}
        schedule_file = config['data'].get('distill_schedule_file')
        if schedule_file:
            if self.method != 'opsd':
                raise ValueError('Distill replay is only supported for the fixed-teacher baseline')
            if file_hash(schedule_file) != config['data']['distill_schedule_sha256']:
                raise ValueError('Distill replay hash mismatch')
            schedule = json.loads(Path(schedule_file).read_text())
            size = self.microbatch * self.accumulation
            if (len(schedule) != train['max_steps'] or any(
                    row['step'] != i + 1 or len(row['ids']) != size
                    for i, row in enumerate(schedule))):
                raise ValueError('Distill replay budget differs from training configuration')
            indices = {e[0]['id']: i for i, e in enumerate(self.prepared['distill'])}
            try:
                order = [indices[key] for row in schedule for key in row['ids']]
            except KeyError as error:
                raise ValueError(f'Replayed example is not eligible: {error}') from error
            self.sampler['distill'] = {'order': order, 'position': 0}
        self.step, self.total_tokens = 0, 0
        (self.output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
        save_json(self.output / 'environment.json', environment())
        save_json(self.output / 'data_manifest.json', self.manifest)
        save_json(self.output / 'coverage.json', {
            split: {'kept': len(values), 'skipped': self.skipped[split],
                    'ids': [e[0]['id'] for e in values]} for split, values in self.prepared.items()})
        save_json(self.output / 'model_metadata.json', {
            'name': config['model']['name'], 'requested_revision': config['model'].get('revision'),
            'model_type': bundle.model.config.model_type,
            'resolved_revision': getattr(bundle.model.config, '_commit_hash', None)})

    def prepare(self, example):
        prompts = self.config.get('prompts', {})
        style = prompts.get('style', 'legacy')
        s = student_prompt(self.tokenizer, example['problem'], style, prompts.get('student_thinking', False))
        t = teacher_prompt(self.tokenizer, example, self.config['data']['privilege'], style,
                           prompts.get('teacher_thinking', True))
        cfg = self.config['rollout']
        if max(len(s), len(t)) + cfg['max_new_tokens'] > cfg['max_total_tokens']:
            return None
        if len(s) + cfg['max_new_tokens'] > cfg.get('max_student_tokens', cfg['max_total_tokens']):
            return None
        return example, s, t

    def next_example(self, split):
        state = self.sampler[split]
        if state['position'] == len(state['order']):
            state['order'] = list(range(len(self.prepared[split])))
            random.shuffle(state['order'])
            state['position'] = 0
        index = state['order'][state['position']]
        state['position'] += 1
        return self.prepared[split][index]

    def rollout_batch(self, examples):
        cfg = self.config['rollout']
        responses, truncated = [], []
        for item, s, t in examples:
            response, cutoff = sample_rollout(self.bundle, self.tokenizer, s, cfg['max_new_tokens'],
                cfg['temperature'], cfg['top_p'], cfg.get('top_k'), cfg.get('min_p', 0.0))
            responses.append(response)
            truncated.append(cutoff)
            self.total_tokens += len(response)
        views = tuple(make_batch_view([e[position] for e in examples], responses,
                      self.bundle.device, self.tokenizer.pad_token_id) for position in (1, 2))
        return views, {'response_tokens': sum(map(len, responses)),
                       'truncated': sum(truncated) / len(examples), 'ids': [e[0]['id'] for e in examples]}

    def _micro_step(self, examples):
        bundle = self.bundle
        options = loss_options(self.config['loss'])
        if self.method == 'sft':
            responses = [self.tokenizer.encode(e[0]['solution'], add_special_tokens=False) +
                         [self.tokenizer.eos_token_id] for e in examples]
            if any(len(e[1]) + len(r) > self.config['rollout']['max_total_tokens'] for e, r in zip(examples, responses)):
                raise ValueError('SFT answer exceeds token budget')
            view = make_batch_view([e[1] for e in examples], responses, bundle.device, self.tokenizer.pad_token_id)
            logits = bundle.forward(view, 'student')
            targets = view['input_ids'][:, view['start'] + 1:view['start'] + 1 + view['length']].clone()
            targets[~view['response_mask']] = -100
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            return loss, sum(map(len, responses)), {'truncated': 0, 'response_tokens': sum(map(len, responses))}
        views, metrics = self.rollout_batch(examples)
        if self.method == 'meta_opsd':
            quiz, quiz_metrics = self.rollout_batch([self.next_example('quiz')])
            gradients, meta_metrics = teacher_meta_grad(bundle, views, quiz,
                self.config['train']['inner_lr'], **options)
            if not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Nonfinite teacher meta-gradient')
            metrics.update(meta_metrics, teacher_meta_grad_norm=norm(gradients),
                           quiz_response_tokens=quiz_metrics['response_tokens'])
            for parameter, gradient in zip(self.teacher.values(), gradients):
                parameter.grad = gradient.detach()
            if self.config['train'].get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(tuple(self.teacher.values()), self.config['train']['max_grad_norm'])
            self.teacher_opt.step()
            self.teacher_opt.zero_grad(set_to_none=True)
            del gradients, quiz
        with torch.no_grad():
            # Official fixed_teacher is the backbone with ALL adapters disabled.
            target = bundle.forward(views[1], 'reference' if self.method == 'opsd' else 'teacher').detach()
        loss = forward_kl(target, bundle.forward(views[0], 'student'),
                          mask=views[0]['response_mask'], **options)
        return loss, int(views[0]['response_mask'].sum()), metrics

    def train_step(self):
        if self.method=='meta_opsd' and (self.microbatch*self.accumulation>1 or
                                        self.config['train'].get('quiz_batch_size',1)>1):
            return self._meta_batch_step()
        started = time.perf_counter()
        self.student_opt.zero_grad(set_to_none=True)
        self.teacher_opt.zero_grad(set_to_none=True)
        total_tokens, weighted_loss, truncated, response_tokens = 0, 0.0, 0.0, 0
        metrics = {'method': self.method}
        ids = []
        for micro_index in range(self.accumulation):
            examples = [self.next_example('distill') for _ in range(self.microbatch)]
            loss, tokens, micro_metrics = self._micro_step(examples)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite student loss')
            # Token weighting exactly matches masked global-token batchmean.
            (loss * tokens).backward()
            total_tokens += tokens
            weighted_loss += loss.item() * tokens
            truncated += micro_metrics['truncated']
            response_tokens += micro_metrics['response_tokens']
            metrics.update(micro_metrics)
            ids.extend(micro_metrics.get('ids', []))
            if self.config['logging'].get('micro_progress', False):
                print(json.dumps({'event':'microbatch', 'optimizer_step':self.step+1,
                    'microbatch':micro_index+1, 'of':self.accumulation,
                    'response_tokens':micro_metrics['response_tokens']}),flush=True)
            del loss
        for p in self.student.values():
            if p.grad is None:
                raise RuntimeError('Missing student gradient')
            p.grad.div_(total_tokens)
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError('Nonfinite student gradient')
        metrics.update(student_loss=weighted_loss / total_tokens,
                       student_grad_norm=norm(p.grad for p in self.student.values()),
                       response_tokens=response_tokens, truncated=truncated / self.accumulation,
                       effective_batch_size=self.microbatch * self.accumulation,
                       ids=ids, micro_batch_size=self.microbatch,
                       gradient_accumulation_steps=self.accumulation,
                       learning_rate=self.student_opt.param_groups[0]['lr'])
        if self.config['train'].get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(tuple(self.student.values()), self.config['train']['max_grad_norm'])
        self.student_opt.step()
        self.scheduler.step()
        self.student_opt.zero_grad(set_to_none=True)
        self.step += 1
        if self.bundle.device.type == 'cuda':
            torch.cuda.synchronize()
            metrics.update(peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                           peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
        metrics.update(step=self.step, seconds=time.perf_counter() - started,
                       cumulative_rollout_tokens=self.total_tokens)
        return metrics

    def _meta_batch_step(self):
        started=time.perf_counter()
        self.student_opt.zero_grad(set_to_none=True)
        self.teacher_opt.zero_grad(set_to_none=True)
        chunks=[];ids=[];response_tokens=0;truncated=0.0
        verbose=self.config['logging'].get('micro_progress',False)
        def progress(phase,index):
            if verbose:
                print(json.dumps({'event':'meta_batch_progress','optimizer_step':self.step+1,
                    'phase':phase,'microbatch':index,'of':self.accumulation}),flush=True)
        # Only token IDs and masks are retained; generation never retains a graph.
        # All trajectories are sampled at the same real student parameters.
        for index in range(self.accumulation):
            examples=[self.next_example('distill') for _ in range(self.microbatch)]
            views,metrics=self.rollout_batch(examples)
            chunks.append(views);ids.extend(metrics['ids'])
            response_tokens+=metrics['response_tokens'];truncated+=metrics['truncated']
            progress('rollout',index+1)
        quiz,quiz_metrics=self.rollout_batch([self.next_example('quiz') for _ in
            range(self.config['train'].get('quiz_batch_size',1))])
        options=loss_options(self.config['loss'])
        gradients,meta_metrics=accumulated_teacher_meta_grad(self.bundle,chunks,quiz,
            self.config['train']['inner_lr'],callback=progress,**options)
        if not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Nonfinite accumulated teacher meta-gradient')
        teacher_norm=norm(gradients)
        for p,g in zip(self.teacher.values(),gradients):p.grad=g.detach()
        del gradients,quiz
        clip=self.config['train'].get('max_grad_norm')
        if clip:torch.nn.utils.clip_grad_norm_(tuple(self.teacher.values()),clip)
        self.teacher_opt.step();self.teacher_opt.zero_grad(set_to_none=True)
        counts=[int(views[0]['response_mask'].sum()) for views in chunks]
        total=sum(counts);weighted_loss=0.0
        for index,(views,count) in enumerate(zip(chunks,counts)):
            with torch.no_grad():target=self.bundle.forward(views[1],'teacher').detach()
            loss=forward_kl(target,self.bundle.forward(views[0],'student'),
                mask=views[0]['response_mask'],**options)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pilot student loss')
            weighted_loss+=loss.item()*count/total
            (loss*(count/total)).backward()
            del target,loss
            progress('pilot',index+1)
        for p in self.student.values():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('Invalid accumulated student gradient')
        student_norm=norm(p.grad for p in self.student.values())
        learning_rate=self.student_opt.param_groups[0]['lr']
        if clip:torch.nn.utils.clip_grad_norm_(tuple(self.student.values()),clip)
        self.student_opt.step();self.scheduler.step()
        self.student_opt.zero_grad(set_to_none=True)
        self.step+=1
        metrics={'method':self.method,**meta_metrics,'ids':ids,'response_tokens':response_tokens,
            'quiz_response_tokens':quiz_metrics['response_tokens'],
            'truncated':truncated/self.accumulation,'student_loss':weighted_loss,
            'student_grad_norm':student_norm,'teacher_meta_grad_norm':teacher_norm,
            'effective_batch_size':self.microbatch*self.accumulation,
            'micro_batch_size':self.microbatch,'gradient_accumulation_steps':self.accumulation,
            'quiz_batch_size':self.config['train'].get('quiz_batch_size',1),
            'learning_rate':learning_rate,'step':self.step,
            'cumulative_rollout_tokens':self.total_tokens}
        if self.bundle.device.type=='cuda':
            torch.cuda.synchronize()
            metrics.update(peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
        metrics['seconds']=time.perf_counter()-started
        return metrics

    def save(self):
        state = {'splits': self.sampler, 'total_tokens': self.total_tokens}
        save_checkpoint(self.output / 'checkpoints' / f'step_{self.step:06d}', self.bundle,
                        self.student_opt, self.teacher_opt, self.step, state, self.config, self.manifest,
                        student_scheduler=self.scheduler)

    def resume(self, path):
        state = torch.load(Path(path) / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['manifest'] != self.manifest or state['config'] != self.config:
            raise ValueError('Resume requires identical data and configuration')
        state = load_checkpoint(path, self.bundle, self.student_opt, self.teacher_opt)
        self.step = state['step']
        self.sampler = state['sampler']['splits']
        self.total_tokens = state['sampler']['total_tokens']
        if state.get('student_scheduler'):
            self.scheduler.load_state_dict(state['student_scheduler'])

    def fit(self):
        with (self.output / 'metrics.jsonl').open('a') as stream:
            while self.step < self.config['train']['max_steps']:
                metrics = self.train_step()
                stream.write(json.dumps(metrics) + '\n')
                stream.flush()
                print(json.dumps(metrics), flush=True)
                if (self.step % self.config['logging']['save_every'] == 0 or
                    (self.step==1 and self.config['logging'].get('save_first_step',False))):
                    self.save()
        self.save()

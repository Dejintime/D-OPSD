import torch
from contextlib import nullcontext

from .losses import forward_kl


def virtual_update(bundle, train_student_view, train_teacher_view, inner_lr, temperature=1.0,
                   token_clip=None, temperature_squared=True):
    context = (bundle.activation_offload()
               if bundle.offload_inner_activations else nullcontext())
    with context:
        return _virtual_update(bundle, train_student_view, train_teacher_view, inner_lr,
                               temperature, token_clip, temperature_squared)


def _virtual_update(bundle, train_student_view, train_teacher_view, inner_lr, temperature,
                    token_clip, temperature_squared):
    student = bundle.parameters("student")
    # Teacher logits must not be detached in the teaching experiment.
    teacher_logits = bundle.forward(train_teacher_view, "teacher")
    student_logits = bundle.forward(train_student_view, "student")
    loss = forward_kl(teacher_logits, student_logits, temperature, token_clip,
                      train_student_view.get("response_mask"), temperature_squared)
    gradients = torch.autograd.grad(loss, tuple(student.values()), create_graph=True)
    fast = {name: value - inner_lr * grad for (name, value), grad in zip(student.items(), gradients)}
    return fast, loss


def teacher_meta_grad(bundle, train_views, quiz_views, inner_lr, temperature=1.0,
                      token_clip=None, temperature_squared=True):
    if bundle.meta_gradient == 'jvp':
        return jvp_meta_grad(bundle, train_views, quiz_views, inner_lr, temperature,
                             token_clip, temperature_squared)
    context = bundle.activation_offload() if bundle.offload_inner_activations else nullcontext()
    with context:
        return _teacher_meta_grad(bundle, train_views, quiz_views, inner_lr, temperature,
                                  token_clip, temperature_squared)


def _teacher_meta_grad(bundle, train_views, quiz_views, inner_lr, temperature,
                       token_clip, temperature_squared):
    train_s, train_t = train_views
    quiz_s, quiz_t = quiz_views
    with torch.no_grad():
        reference_logits = bundle.forward(quiz_t, "reference").detach()
        before = forward_kl(reference_logits, bundle.forward(quiz_s, "student"), temperature,
                            mask=quiz_s.get("response_mask"), temperature_squared=temperature_squared)
    fast, inner_loss = virtual_update(bundle, train_s, train_t, inner_lr, temperature,
                                      token_clip, temperature_squared)
    after = forward_kl(reference_logits, bundle.forward(quiz_s, "student", fast), temperature,
                       mask=quiz_s.get("response_mask"), temperature_squared=temperature_squared)
    teacher = bundle.parameters("teacher")
    gradients = torch.autograd.grad(after, tuple(teacher.values()))
    # Only teacher gradients are returned; real student parameters are untouched.
    metrics = {"inner_loss": inner_loss.item(), "quiz_before": before.item(),
               "quiz_after": after.item(), "learning_progress": (before - after).item()}
    return gradients, metrics


def jvp_meta_grad(bundle, train_views, quiz_views, inner_lr, temperature=1.0,
                   token_clip=None, temperature_squared=True):
    """Exact chain rule for one SGD inner step; no full reverse-over-reverse graph.

    v = d Lquiz / d theta_fast.  Hypergradient = -alpha * mixed_Hessian @ v.
    J_student @ v is a forward-mode directional derivative. The remaining
    mixed derivative is an ordinary teacher backward through a weighted softmax.
    Pointwise clip's mask has zero derivative away from its boundary, matching
    autograd's piecewise clamp derivative (including the unclipped equality).
    """
    train_s, train_t=train_views
    quiz_s, quiz_t=quiz_views
    student=bundle.parameters('student')
    with torch.no_grad():
        reference=bundle.forward(quiz_t,'reference').detach()
        before=forward_kl(reference,bundle.forward(quiz_s,'student'),temperature,
            mask=quiz_s.get('response_mask'),temperature_squared=temperature_squared)
        target=bundle.forward(train_t,'teacher').detach()
    logits=bundle.forward(train_s,'student')
    inner_loss=forward_kl(target,logits,temperature,token_clip,
                          train_s.get('response_mask'),temperature_squared)
    inner_value=inner_loss.item()
    g=torch.autograd.grad(inner_loss,tuple(student.values()))
    fast={name:(p-inner_lr*grad).detach().requires_grad_(True)
          for (name,p),grad in zip(student.items(),g)}
    del g,logits,target,inner_loss
    after=forward_kl(reference,bundle.forward(quiz_s,'student',fast),temperature,
                     mask=quiz_s.get('response_mask'),temperature_squared=temperature_squared)
    after_value=after.item()
    values=torch.autograd.grad(after,tuple(fast.values()))
    direction=dict(zip(student,values))
    del fast,after,reference
    primal,tangent=bundle.student_jvp(train_s,direction)
    del direction,values
    precision=torch.float64 if primal.dtype==torch.float64 else torch.float32
    log_s=torch.log_softmax(primal.to(precision)/temperature,-1).detach()
    tangent=tangent.to(precision).detach()
    teacher_logits=bundle.forward(train_t,'teacher')
    log_t=torch.log_softmax(teacher_logits.to(precision)/temperature,-1)
    probability=log_t.exp()
    with torch.no_grad():
        mask=torch.ones_like(probability) if token_clip is None else (
            probability.detach()*(log_t.detach()-log_s) <= token_clip).to(precision)
        coefficient=(log_s.exp()*tangent).sum(-1,keepdim=True)-tangent
        coefficient=coefficient*mask
    per_token=(probability*coefficient).sum(-1)
    valid=train_s.get('response_mask')
    objective=per_token.mean() if valid is None else per_token.masked_select(valid).mean()
    scale=(temperature**2 if temperature_squared else 1.0)/temperature
    gradients=torch.autograd.grad(-inner_lr*scale*objective,tuple(bundle.parameters('teacher').values()))
    return gradients,{'inner_loss':inner_value,'quiz_before':before.item(),'quiz_after':after_value,
                      'learning_progress':before.item()-after_value,'meta_gradient':'exact_one_step_jvp'}


def _micro_hypergradient(bundle,views,direction,inner_lr,temperature=1.0,
                        token_clip=None,temperature_squared=True):
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
    objective=per_token.masked_select(views[0]['response_mask']).mean()
    scale=(temperature**2 if temperature_squared else 1.0)/temperature
    return torch.autograd.grad(-inner_lr*scale*objective,tuple(bundle.parameters('teacher').values()))


def accumulated_teacher_meta_grad(bundle,chunks,quiz,inner_lr,callback=None,**options):
    """One shared SGD inner step over token-weighted microbatches; exact JVP feedback.

    No graph crosses microbatch boundaries. Inner gradients are accumulated
    numerically, then the mixed derivative is recomputed with one fixed Quiz
    gradient at the shared fast parameters. Rollout tokens must remain fixed.
    """
    if not chunks:raise ValueError('No training microbatches')
    student=bundle.parameters('student')
    counts=[int(views[0]['response_mask'].sum()) for views in chunks]
    total=sum(counts)
    if min(counts)<1:raise ValueError('Empty training continuation')
    weights=[n/total for n in counts]
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
            for value,g in zip(accumulated,gradients):value.add_(g.detach(),alpha=weight)
        del target,logits,loss,gradients
        if callback:callback('inner',index+1)
    fast={name:(p-inner_lr*g).detach().requires_grad_(True)
          for (name,p),g in zip(student.items(),accumulated)}
    del accumulated
    quiz_options={k:v for k,v in options.items() if k!='token_clip'}
    with torch.no_grad():
        reference=bundle.forward(quiz[1],'reference').detach()
        before=forward_kl(reference,bundle.forward(quiz[0],'student'),
            mask=quiz[0]['response_mask'],**quiz_options).item()
    after=forward_kl(reference,bundle.forward(quiz[0],'student',fast),
        mask=quiz[0]['response_mask'],**quiz_options)
    after_value=after.item()
    values=torch.autograd.grad(after,tuple(fast.values()))
    direction=dict(zip(student,values))
    del fast,reference,after,values
    accumulated=None
    for index,(views,weight) in enumerate(zip(chunks,weights)):
        gradients=_micro_hypergradient(bundle,views,direction,inner_lr,**options)
        if accumulated is None:accumulated=[g.detach()*weight for g in gradients]
        else:
            for value,g in zip(accumulated,gradients):value.add_(g.detach(),alpha=weight)
        del gradients
        if callback:callback('meta',index+1)
    return accumulated,{'inner_loss':inner_value,'quiz_before':before,'quiz_after':after_value,
        'learning_progress':before-after_value,'meta_gradient':'exact_shared_inner_batch_jvp'}

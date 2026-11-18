import torch.nn.functional as F


def forward_kl(teacher_logits, student_logits, temperature=1.0, token_clip=None,
               mask=None, temperature_squared=True):
    """Full forward KL over vocabulary; one response per view; no prompt tokens."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher/student continuation logits must align")
    # Preserve FP64 for finite-difference tests; upcast low precision for training.
    t = teacher_logits if teacher_logits.dtype.is_floating_point and teacher_logits.element_size() >= 4 else teacher_logits.float()
    s = student_logits if student_logits.dtype.is_floating_point and student_logits.element_size() >= 4 else student_logits.float()
    log_t = F.log_softmax(t / temperature, dim=-1)
    log_s = F.log_softmax(s / temperature, dim=-1)
    contributions = log_t.exp() * (log_t - log_s)
    if token_clip is not None:
        contributions = contributions.clamp(max=token_clip)
    per_token = contributions.sum(-1)
    if mask is None:
        loss = per_token.mean()
    else:
        if mask.shape != per_token.shape or not mask.any():
            raise ValueError("Invalid continuation mask")
        loss = per_token.masked_select(mask).mean()
    return loss * (temperature ** 2 if temperature_squared else 1.0)


def loss_options(config):
    return {"temperature": config.get("temperature", 1.0),
            "token_clip": config.get("token_clip"),
            "temperature_squared": config.get("temperature_squared", True)}

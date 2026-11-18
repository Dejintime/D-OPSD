import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def environment():
    import transformers
    import peft
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def norm(tensors):
    tensors = list(tensors)
    return torch.stack([t.detach().float().square().sum() for t in tensors]).sum().sqrt().item()


def save_checkpoint(path, bundle, student_opt, teacher_opt, step, sampler, config, manifest,
                    student_scheduler=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(path / "adapters", selected_adapters=["student", "teacher"])
    adapter_params = {name: p.detach().cpu() for role in ("student", "teacher")
                      for name, p in bundle.parameters(role).items()}
    state = {"adapters": adapter_params, "student_optimizer": student_opt.state_dict(),
             "teacher_optimizer": teacher_opt.state_dict(), "step": step, "sampler": sampler,
             "config": config, "manifest": manifest,
             "student_scheduler": student_scheduler.state_dict() if student_scheduler is not None else None,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path / "training_state.tmp"
    torch.save(state, temporary)
    temporary.replace(path / "training_state.pt")


def load_checkpoint(path, bundle, student_opt=None, teacher_opt=None):
    # Only load local checkpoints produced by this project, never untrusted pickle.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    named = dict(bundle.model.named_parameters())
    with torch.no_grad():
        for name, tensor in state["adapters"].items():
            named[name].copy_(tensor)
    if student_opt is not None:
        student_opt.load_state_dict(state["student_optimizer"])
        teacher_opt.load_state_dict(state["teacher_optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state

"""Offline end-to-end engineering check. Random tiny weights are NOT a benchmark."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from dual_opsd.data import file_hash, write_jsonl
from dual_opsd.models import ModelBundle
from dual_opsd.trainer import Trainer
from dual_opsd.utils import load_config, save_json, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--opsd-settings", action="store_true")
    parser.add_argument('--micro-batch-size',type=int,default=1)
    parser.add_argument('--gradient-accumulation-steps',type=int,default=1)
    args = parser.parse_args()
    seed_all(42)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "Problem": 3, "1": 4,
             "2": 5, "3": 6, "+": 7, "=": 8, "answer": 9, "boxed": 10}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + '\n' }}{% endfor %}"
    model_config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               eos_token_id=2, pad_token_id=1, max_position_embeddings=1024)
    model_config._attn_implementation = "eager"
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    bundle = ModelBundle(Qwen2ForCausalLM(model_config).to(device=args.device, dtype=dtype), rank=2, alpha=2)
    for role in ("student", "teacher"):
        for p in bundle.parameters(role).values():
            p.data = p.data.float()
    with tempfile.TemporaryDirectory(prefix="opsd-smoke-") as directory:
        root = Path(directory)
        rows = [{"id": "synthetic/0", "problem": "1 + 1 =", "solution": "1 + 1 = 2", "answer": "2"}]
        for split in ("distill", "quiz"):
            write_jsonl(root / f"{split}.jsonl", [dict(rows[0], id=f"synthetic/{split}")])
        save_json(root / "manifest.json", {"synthetic": True,
                  "hashes": {s: file_hash(root / f"{s}.jsonl") for s in ("distill", "quiz")}})
        config = load_config("configs/legacy_qwen25/debug_3090.yaml")
        config["model"].update(name="RANDOM_TINY_SMOKE_ONLY", device=args.device)
        config["data"]["root"] = str(root)
        config["rollout"].update(max_new_tokens=4, max_total_tokens=256)
        config["train"].update(inner_lr=0.1, student_lr=0.1, teacher_lr=0.001)
        config['train'].update(micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps)
        if args.micro_batch_size*args.gradient_accumulation_steps>1:
            bundle.meta_gradient='jvp'
        if args.opsd_settings:
            config['train'].update(student_optimizer='adamw', student_lr=0.001,
                                    lr_scheduler='linear', max_grad_norm=0.1)
            config['loss'].update(kind='clipped_forward_kl', temperature=1.1,
                                  token_clip=0.05, temperature_squared=False)
        trainer = Trainer(config, bundle, tokenizer, root / "meta_run")
        backbone_before = {n: p.detach().clone() for n, p in bundle.model.named_parameters() if "lora_" not in n}
        student_before = {n: p.detach().clone() for n, p in bundle.parameters("student").items()}
        teacher_before = {n: p.detach().clone() for n, p in bundle.parameters("teacher").items()}
        metrics = trainer.train_step()
        assert metrics["teacher_meta_grad_norm"] > 0, metrics
        assert any(not torch.equal(p, teacher_before[n]) for n, p in bundle.parameters("teacher").items())
        assert any(not torch.equal(p, student_before[n]) for n, p in bundle.parameters("student").items())
        for n, p in bundle.model.named_parameters():
            if n in backbone_before:
                assert torch.equal(p, backbone_before[n])
        trainer.save()
        checkpoint = root / "meta_run/checkpoints/step_000001"
        # Compare uninterrupted and restored next steps, including rollout RNG.
        uninterrupted = trainer.train_step()
        expected = {n: p.detach().clone() for n, p in bundle.model.named_parameters() if "lora_" in n}
        trainer.resume(checkpoint)
        restored = trainer.train_step()
        assert uninterrupted["student_loss"] == restored["student_loss"]
        for n, p in bundle.model.named_parameters():
            if n in expected:
                torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
        fixed_config = copy.deepcopy(config)
        fixed_config["experiment"]["method"] = "opsd"
        if args.opsd_settings:
            fixed_config['train'].update(micro_batch_size=2, gradient_accumulation_steps=2,
                                          gradient_checkpointing=True)
        fixed = Trainer(fixed_config, bundle, tokenizer, root / "fixed_run")
        teacher_before = {n: p.detach().clone() for n, p in bundle.parameters("teacher").items()}
        fixed.train_step()
        for n, p in bundle.parameters("teacher").items():
            assert torch.equal(p, teacher_before[n])
        sft_config = copy.deepcopy(config)
        sft_config["experiment"]["method"] = "sft"
        Trainer(sft_config, bundle, tokenizer, root / "sft_run").train_step()
        from dual_opsd.evaluation import evaluate_student
        evaluation = evaluate_student(bundle, tokenizer, rows, root / "evaluation", max_new_tokens=4)
        assert evaluation["n"] == 1
        print({"status": "passed", "device": args.device, "dtype": str(dtype), "meta_step": metrics,
               "checks": ["nonzero teacher gradient", "both adapters updated", "backbone frozen",
                          "exact checkpoint resume", "fixed teacher invariant", "sft step", "evaluation"]})


if __name__ == "__main__":
    main()

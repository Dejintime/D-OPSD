"""Check CUDA and optionally vLLM using tiny random weights, without downloads."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--vllm-smoke',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args()
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import datasets,peft,transformers,math_verify,vllm
    report={'versions':{name:importlib.metadata.version(name) for name in
        ['torch','transformers','peft','datasets','vllm','math-verify','pyarrow']},'gpus':[],
        'TRITON_LIBCUDA_PATH':os.environ.get('TRITON_LIBCUDA_PATH')}
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():raise RuntimeError(f'GPU {index} lacks BF16 support')
            tensor=torch.randn(64,64,device=f'cuda:{index}',dtype=torch.bfloat16)
            assert torch.isfinite(tensor@tensor.T).all()
            torch.cuda.synchronize(index)
            report['gpus'].append({'index':index,'name':torch.cuda.get_device_name(index),
                'bf16_matmul':'passed'})
            del tensor
            torch.cuda.empty_cache()
    if args.vllm_smoke:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
        from peft import LoraConfig,get_peft_model
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        with tempfile.TemporaryDirectory(prefix='opsd-env-check-') as directory:
            root=Path(directory)
            config=Qwen3Config(vocab_size=64,hidden_size=128,intermediate_size=256,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                head_dim=64,max_position_embeddings=128,eos_token_id=1,pad_token_id=0,
                tie_word_embeddings=True)
            model=Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
            model.save_pretrained(root/'model')
            vocab={'[PAD]':0,'[EOS]':1,'[UNK]':2,**{f'token{i}':i for i in range(3,64)}}
            backend=Tokenizer(WordLevel(vocab=vocab,unk_token='[UNK]'))
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',
                eos_token='[EOS]',unk_token='[UNK]')
            tokenizer.save_pretrained(root/'model')
            adapter=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,task_type='CAUSAL_LM',
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),
                adapter_name='student')
            adapter.save_pretrained(root/'adapters',selected_adapters=['student'])
            llm=LLM(model=str(root/'model'),dtype='bfloat16',max_model_len=128,
                gpu_memory_utilization=0.10,tensor_parallel_size=1,distributed_executor_backend='mp',
                max_num_seqs=2,max_num_batched_tokens=128,enforce_eager=True,
                enable_lora=True,max_lora_rank=64,max_loras=1,max_cpu_loras=1,
                generation_config='vllm',seed=42)
            try:
                params=SamplingParams(temperature=1.0,top_p=0.95,top_k=-1,max_tokens=4,seed=42)
                prompts=[{'prompt_token_ids':[3,4]}]
                base=llm.generate(prompts,params,use_tqdm=False)
                tuned=llm.generate(prompts,params,use_tqdm=False,
                    lora_request=LoRARequest('student',1,str(root/'adapters/student')))
                for outputs in [base,tuned]:
                    assert len(outputs)==1 and 0<len(outputs[0].outputs[0].token_ids)<=4
                report['vllm_tiny_qwen3_base_and_lora']='passed'
            finally:
                llm.llm_engine.engine_core.shutdown()
    report['status']='passed'
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

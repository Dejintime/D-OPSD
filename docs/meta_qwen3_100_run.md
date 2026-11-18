# Qwen3 双 LoRA：100 步试跑

最新评测已切换为 AIME24、AIME25，每题 12 次、输出上限 38912；当前只运行这两套 benchmark，详情见 [AIME 评测说明](aime_evaluation.md)。下文的 MATH-500 段落保留为已完成实验的记录，不再执行。

本轮目录：`/home/yangdejin/OPSD/outputs/meta_qwen3_100_20261111_v1`。

## 训练

从用户的 `model/Qwen3-1.7B` 初始权重重新开始，使用 `configs/meta_qwen3_100_run.yaml`。有效 batch=1，Quiz batch=1，累积步数=1，共 100 次参数更新；每 25 步保存学生、教师和优化器状态。

两套 LoRA 分别是学生策略和可学习的特权教师，共享冻结骨干。学生真实更新使用 AdamW，学习率从 5e-6 线性降至零；教师 AdamW 学习率 1e-5。元梯度对应一步 SGD 内层模拟，inner_lr=1e-4，使用数值核对过的 JVP 实现。它不是对真实 AdamW 更新求得的精确超梯度。

训练数据为 `Openthoughts_math_30k_opsd` 的独立 distill/quiz 划分。学生训练时关闭 thinking，教师开启；训练采样上限 1024，temperature=1.1，top_p=0.95，top_k=20。LoRA rank=64、alpha=128、七类投影。上下文与 batch 等单卡适配项不代表官方全配置复现。

## 前后评测

按用户最新要求，改为完整 MATH-500：每题生成 1 次、thinking 开启、每次最多 4096 个输出 token；训练前后各 500 条回答。数据文件为 `data/processed/math/test.jsonl`，其 MATH-500 revision 为 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`，完整 500 题的文件 hash 记录在 metadata。

新的评测目录为 `outputs/meta_qwen3_100_20261111_v1/math500_4k_s1`。之前的 OpenThoughts dev 50 题、12 次采样、38912-token 评测已停止，已有记录保留，不混入本轮 MATH-500 比较。

使用独立 `.venv_eval` 的 vLLM 0.11.0 / torch 2.8.0 / transformers 4.57.1 / math-verify 0.8.0，不改变训练环境。temperature=1.0、top_p=0.95、top_k 禁用。使用官方评测提示和最后一个 boxed 答案判分。

同一个 vLLM 引擎交错生成原始模型和第 100 步学生 LoRA 的回答；推理不加载教师 LoRA。相同题目、相同样本编号使用相同派生种子，规则记录在 evaluation metadata。兼容性检查已验证原始模型和 rank-64 学生适配器均可生成。

`scripts/evaluate_vllm_pair.py` 在第 100 步保存且显存释放后启动评测。每完成一个样本即刷盘保存；重新运行相同命令时读取已有记录继续，配置或权重不一致则拒绝混用。每次启动先校验全部 gold 答案与上下文预算，货币答案中的转义美元符号不作为数学模式分隔符。

指标为完整 MATH-500 单次采样正确率，同时报告 4096-token 截断率和答案提取失败率。该解码预算由用户指定，不等同于原 OPSD 论文的长回答、多次采样协议；训练前后比较也不足以单独证明元学习优于固定教师，需要匹配预算的固定教师训练对照。

## 查看与恢复

```bash
cd /home/yangdejin/OPSD
tail -n 1 outputs/meta_qwen3_100_20261111_v1/training/metrics.jsonl
cat outputs/meta_qwen3_100_20261111_v1/math500_4k_s1/vllm_state.json
tail -n 20 outputs/meta_qwen3_100_20261111_v1/math500_4k_s1/vllm_pair.log
```

评测进程停止后，可在项目目录用以下命令续跑；不要与现存评测进程重复启动：

```bash
.venv_eval/bin/python -u scripts/evaluate_vllm_pair.py \
  --root outputs/meta_qwen3_100_20261111_v1 \
  --output-root outputs/meta_qwen3_100_20261111_v1/math500_4k_s1 \
  --data-file data/processed/math/test.jsonl --dataset-label MATH-500 \
  --limit 0 --samples-per-prompt 1 --max-new-tokens 4096 \
  --max-model-len 8192 --max-num-seqs 24
```

在新的评测目录中产生 `baseline_vllm/summary.json`、`trained_vllm/summary.json` 和 `comparison.json`。训练指标与最终准确率应分开解读；Quiz 的一步模拟损失变化不能替代测试集正确率。

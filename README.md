# 双路 OPSD

默认模型为服务器上的 `model/Qwen3-1.7B`，默认数据为 `siyanzhao/Openthoughts_math_30k_opsd`。包含官方风格的固定教师 OPSD 基线，以及独立的双 LoRA 元学习实验分支。上轮 Qwen2.5 配置保存在 `configs/legacy_qwen25/`，结果未覆盖。

## 环境

```bash
source /home/yangdejin/miniconda3/etc/profile.d/conda.sh
conda activate opsd
cd /home/yangdejin/OPSD
python -m pytest -q
```

新服务器 `222.19.197.47` 使用独立 conda 环境 `opsd`（Python 3.12），训练与评测共用 PyTorch 2.8.0、Transformers 4.57.1、PEFT 0.19.1、vLLM 0.11.0。安装约束见 `requirements-opsd.txt`，使用说明见 [新服务器环境与数据](docs/new_server_setup.md)。`requirements-lock.txt` 保留旧服务器训练环境记录，不与新环境约束混装。模型目录为 `model/Qwen3-1.7B`；本轮环境验证使用微型随机权重，正式运行前须确认模型完整分片均已下载结束。

## 数据

官方训练数据的两个 Parquet 分片已保存于 `data/raw/openthoughts/`，固定 revision 为 `1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62`，共 29434 条。新服务器已同步上一轮相同的处理结果，无需重新划分；若需要从原始数据重建到新目录，可运行：

```bash
python scripts/prepare_openthoughts.py \
  --test-file data/processed/openthoughts/test.jsonl \
  --output data/processed/openthoughts_REBUILT
```

输出位于 `data/processed/openthoughts/`。为了支持研究中的独立 Quiz 和开发集，按题目划分为 27925 道蒸馏、1000 道 Quiz、500 道开发题；原 MATH-500 作为独立测试文件复用。排除 9 条重复或与测试重叠记录，具体原因、源文件 hash 与划分 hash 均在 manifest.json 中。

这与原作者直接使用完整训练 split 有区别，不应声称逐项复现官方数据设置。固定教师和元学习比较应使用相同蒸馏划分，并检查各自长度过滤覆盖率。没有将旧 MATH 训练数据混入新蒸馏集。

当前数学 benchmark 为 `data/processed/aime/aime24.jsonl` 和 `aime25.jsonl`，各 30 题，原始 Parquet、revision 与文件 hash 均已保留。后续仅运行 AIME24/25；MATH-500 文件作为已有数据 manifest 的组成部分保留，不自动启动评测。

## 固定教师 OPSD 基线

本轮已完成的 batch=32 双路实验请使用 [匹配基线说明](docs/opsd_baseline.md) 和 `scripts/run_opsd_baseline.py`。它从原运行配置生成 `configs/opsd_qwen3_b32_100.yaml` 对应的设置，重放同一3200条蒸馏题序，支持训练、AIME24/25评测和双路对照汇总。下述旧配置分别用于官方风格参考和早期实验。

配置：`configs/opsd_qwen3.yaml`。共享 GPU 的保守默认值为 microbatch 1、累积 32；`configs/opsd_qwen3_idle.yaml` 保留空闲 GPU 下实测通过的 microbatch 2、累积 16。主要设置为：

- rank 64、alpha 128，q/k/v/o/gate/up/down 七类投影。
- AdamW，lr 5e-6，betas 0.9/0.999，epsilon 1e-8，weight decay 0；线性学习率调度、梯度范数裁剪 0.1。
- 每题一条 rollout，最多 1024 个输出 token，temperature 1.1、top_p 0.95、top_k 20。
- 学生关闭 thinking，教师开启 thinking，使用官方 reason_first=False 提示结构。
- 全词表 forward KL，温度 1.1，逐词表项贡献裁剪上限 0.05，不额外乘温度平方。
- 有效 batch 32，100 个 optimizer steps，每 25 步保存。单卡通过 microbatch 与梯度累积实现；梯度按有效 token 数加权，使结果等价于全 batch 的 token 平均。
- 固定教师前向关闭全部 LoRA，权重始终是初始骨干。只有学生 LoRA 更新。

单卡使用 PyTorch SDPA 和 non-reentrant gradient checkpointing，替代官方的 FlashAttention2/vLLM 多卡实现。当前 rollout 按题顺序生成，教师评分和学生反向按 microbatch 批处理；因此大 microbatch 主要提升评分与反向效率，不代表生成阶段也按 batch 并行。

```bash
python scripts/tune_batch.py --config configs/opsd_qwen3.yaml \
  --output outputs/qwen3_batch_scan_NEW
python scripts/train.py --config configs/opsd_qwen3.yaml
```

batch 校准先用实际数据的最大提示长度与固定 1024-token 人工续写进行显存压力测试，再用真实采样完成一次有效 batch 32 的参数更新，保存 checkpoint。压力测试不用于报告准确率。32 个样本的一次更新不等于完整 100 步训练，更不等于完整遍历训练集。

`max_total_tokens=20000` 是总长度资格上限，不预分配 20000-token 张量。实际占用由题目、参考解答和输出共同决定。只对有效输出 token 做损失，并仅对所需输出位置计算词表 logits，以减少无用显存。批内 padding 位置被正确屏蔽。

## 双 LoRA 元学习分支

配置：`configs/meta_qwen3.yaml`。共享冻结 Qwen3 骨干，student 与 teacher 使用独立 LoRA。临时学生通过一步可微 SGD 更新，教师根据临时学生在独立 Quiz 的完整 KL 接收元梯度，再用更新后的教师对原始真实学生执行 pilot 更新。

该分支保留官方 LoRA、采样和蒸馏设置，但新增 inner_lr=1e-4、teacher_lr=1e-5；真实学生为 AdamW 5e-6，模拟学生为一步 SGD，因此存在明确的内层近似，需要后续消融。Quiz 目标不做 pointwise clipping，蒸馏目标做 clipping。

现已支持 microbatch 和梯度累积：先在同一学生参数下收集全部轨迹，以全局有效 token 数加权得到一次共享内层 SGD 更新，再对同一个 Quiz 目标累积精确教师元梯度，最后使用更新后的教师执行真实学生更新；不是将多个独立 episode 的元梯度求平均。`configs/meta_qwen3_b32_100.yaml` 使用单卡 micro=2、累积=16、有效 batch=32，原单题配置保留。资源测试见 [容量报告](docs/dual_capacity_report.md)，正式比较仍需匹配样本、token 和计算预算。

默认使用一步 SGD 元梯度的等价 JVP 计算：先计算临时学生数值和 Quiz 梯度，再将学生 logits 的方向导数与教师的一阶反向组合，得到相同的教师超梯度；不用保存完整的反向套反向 Hessian 图。无裁剪/有裁剪目标均已在小模型上与完整展开逐参数核对，CUDA 上也核对了梯度方向。

常规梯度使用 SDPA；仅 JVP 前向使用支持 forward AD 的 eager attention。旧的完整展开仍可通过 model.meta_gradient=unroll 使用。元分支关闭共享骨干的 gradient checkpointing，避免重算时适配器和临时参数切换导致错误。

元分支另启用 CPU 中间张量暂存。常驻模型参数继续使用 GPU 上的原存储，广播视图先压缩再传输，CPU 使用普通缓冲而非大量锁页缓存；真实 GPU 小模型已检查暂存前后的元梯度一致性。启用前要求至少 24 GiB 可用主机内存。不要将固定教师基线的显存结论直接套到这个分支。

```bash
python scripts/profile_step.py --config configs/meta_qwen3.yaml \
  --steps 1 --warmup 0 --output outputs/meta_profile_NEW
```

单卡元学习配置对总长度另设 4096、学生总长 2048 的保护上限；官方固定教师配置没有这项学生限制。保存 coverage.json，不能将两个过滤后样本池差异忽略为算法收益。

`configs/opsd_qwen3_matched.yaml` 是为元学习准备的受控固定教师对照：使用相同单题 batch、长度和数据筛选规则。它与 meta_qwen3 的学生题次/更新次数可以匹配，但有效 batch 1 的设定不同于官方 batch 32；官方参考基线和受控对照应分别报告。

## 评测与恢复

主配置保留论文风格的 thinking=True、最大输出 38912、temperature=1.0、top_p=0.95、关闭 top_k、每题 12 次采样。accuracy 为所有完成样本正确率的平均，即 Avg@12，非 pass@12。评测成本高，训练/显存校准不会自动启动完整评测。

```bash
python scripts/evaluate.py --config configs/opsd_qwen3.yaml --initial --split dev
python scripts/evaluate.py --run outputs/RUN_ID --split dev
python scripts/train.py --config outputs/RUN_ID/config.yaml \
  --output outputs/RUN_ID --resume outputs/RUN_ID/checkpoints/step_000025
```

checkpoint 包括两套 LoRA、优化器、学生 scheduler、采样位置与随机状态。恢复要求数据和配置一致。旧 checkpoint 请使用其原始保存配置；仅加载可信 training_state.pt。

## 历史与来源

- [官方参数对照](docs/opsd_reference.md)
- [Qwen2.5 历史试跑](docs/pilot_run.md)，其中 24% → 30% 不属于 Qwen3。
- [OPSD 固定版本](https://github.com/siyan-zhao/OPSD/tree/ae7d2519e94920c4eb6206c0c26de46d9c50abae)
- [论文 v3](https://arxiv.org/html/2601.18734v3)

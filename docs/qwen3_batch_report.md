# Qwen3 参数切换与 batch 实测

测试日期为 2026 年 11 月 11 日。模型使用用户提供的 `/home/yangdejin/OPSD/model/Qwen3-1.7B`，配置额外检查 model_type=qwen3，防止误加载旧 Qwen2.5。

## 已对齐的训练参数

固定教师基线采用官方 rank 64、alpha 128、q/k/v/o/gate/up/down 七类投影；AdamW 5e-6、线性学习率、max_grad_norm=0.1；每题一次采样、输出上限 1024、temperature=1.1、top_p=0.95、top_k=20。全词表 forward KL 的每个词表条目贡献上限 0.05，softmax 温度 1.1，不乘额外的温度平方。学生 thinking 关闭、教师开启。

单卡使用 SDPA 和非重入 gradient checkpointing，rollout 按题顺序生成，评分与反向按 microbatch 批处理。固定教师就是关闭全部适配器的初始骨干。

## 新数据

官方源数据两个分片共 29434 条，完整保存在 `data/raw/openthoughts`。固定 revision：`1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62`。

研究划分位于 `data/processed/openthoughts`，蒸馏 27925、独立 Quiz 1000、开发 500；测试文件沿用 MATH-500 的 500 题。删除了 9 条重复题；本次精确去空白匹配没有发现 MATH-500 重叠，不代表完成语义近重复或预训练污染审计。源文件、各划分 hash 和删除 ID 均已保存。

这套研究划分为元学习额外留出了 Quiz/dev，不是官方完整 train split 的逐项复现。旧 MATH 数据和 Qwen2.5 结果均保留。

## 固定教师显存扫描

显存扫描使用实际数据的最大提示长度，学生总长 2795、教师总长 4328，并使用人工固定的 1024-token 续写做前向、反向、裁剪和 AdamW 更新。扫描仅用于压力测试，不用于准确率或真实生成速度报告。

| Microbatch | 峰值 allocated | 峰值 reserved | 结果 |
| --- | ---: | ---: | --- |
| 1 | 8.70 GiB | 9.57 GiB | 通过 |
| 2 | 13.67 GiB | 14.79 GiB | 通过 |
| 4 | 超出显存 | — | OOM |

GPU 空闲时，建议 microbatch=2、gradient_accumulation_steps=16，有效 batch=32。服务器测试后又出现约 11 GiB 的其他用户 GPU 占用，因此当前主配置保守设为 microbatch=1、accumulation=32；空闲方案保存在 opsd_qwen3_idle.yaml。

## 真实采样更新

用 microbatch 2、累积 16，完成了一次真实 on-policy 有效 batch 32 的学生参数更新，并保存 checkpoint。

- 32 个学生 rollout，共 26501 个生成 token。
- 完整更新耗时 504.86 秒，即约 8 分 25 秒。
- 峰值 allocated 13.60 GiB，reserved 17.47 GiB。
- clipped forward KL loss 约 0.005983，裁剪前梯度范数约 0.1526，随后按 0.1 裁剪。
- 19/32 条 rollout 达到 1024-token 上限，截断率 59.4%。

这是一次 optimizer update，不是完整 100 步，也不是一个数据集 epoch。仅按这一次长度分布外推，100 个有效 batch-32 更新约需 14 小时；共享 GPU、生成长度和采样结果会改变实际耗时。本轮没有声称获得新的准确率提升。

服务器记录位于 `outputs/qwen3_batch_scan_v1`，checkpoint 为 `checkpoints/step_000001`；推荐配置和真实更新指标分别在 recommendation.json 与 real_update.json。

## 元学习分支

元学习配置保留双 LoRA 与独立 Quiz。由于完整高阶反向图会超出 3090 与主机内存预算，新增针对一步 SGD 内层更新的等价 JVP 超梯度实现。无裁剪、有裁剪两种设置均已与原完整展开逐参数核对；CUDA 测试同时核对了 Flash/SDPA 与 CPU 中间张量暂存组合的梯度一致性。

元学习新增的 inner_lr=1e-4、teacher_lr=1e-5 不来自原 OPSD。真实学生使用 AdamW，而模拟学生仍是一步 SGD，属于明确的内层近似。默认元学习 batch=1，并提供相同 batch 和长度过滤的 opsd_qwen3_matched.yaml 固定教师对照；不要直接将它与有效 batch 32 的基线按相同更新步数比较。

元学习真实模型验证记录单独保存，不能用固定教师的显存数替代其峰值。CPU 中间张量暂存不复制常驻参数，且使用普通 CPU 缓冲，启用前要求至少 24 GiB 可用主机内存。

真实模型元学习 batch 1 的一次更新已通过，记录在 `outputs/qwen3_meta_profile_v5`：完整更新约 55.73 秒，峰值 allocated 8.97 GiB、reserved 10.08 GiB。测试时同卡有其他用户约 11 GiB 的占用，因此耗时不代表独占 GPU 性能；不应直接与基线的一次 batch-32 更新比较速度。

可恢复的元学习验证另存于 `outputs/qwen3_meta_validation_v1/checkpoints/step_000001`，包含实际更新后的教师/学生 LoRA、两个优化器和 scheduler 状态；该次更新约 53.61 秒，峰值 allocated 同为 8.97 GiB。尚未运行新 Qwen3 的完整 100 步或训练后准确率评测。

## 一步 SGD 元梯度的计算校验

JVP 只是改变计算方法，没有把一步 SGD 模拟更新的超梯度替换成一阶近似。令临时学生为 theta_fast=theta-alpha*g，v 为 Quiz 损失对 theta_fast 的梯度，则教师超梯度是 -alpha 乘以 mixed-Hessian 与 v 的乘积。

实现先算学生 logits 对参数方向 v 的导数 s，再通过教师 softmax 的一次反向得到同一个混合导数。对温度 T、逐词表条目裁剪 mask m 和学生分布 pS，每个位置的教师辅助目标为 sum(m*pT*(sum(pS*s)-s))/T，再乘 -alpha；按有效输出位置平均。mask 在裁剪边界外的导数为零，与原 clamp 目标的局部导数一致。启用温度平方缩放时再乘 T²，官方配置不启用。

FP64 小模型测试同时覆盖有裁剪和无裁剪，逐参数比较完整展开与 JVP 超梯度；CUDA 测试覆盖不同 attention 后端与中间张量暂存。该等价性针对当前独立教师/学生 LoRA、冻结骨干和一步 SGD 内层更新；不宣称它等价于对真实 AdamW 更新求超梯度。

# OPSD 官方参数与当前实现差异

核对时间：2026 年 11 月 11 日。官方仓库为 siyan-zhao/OPSD，本次查询 HEAD 为 `ae7d2519e94920c4eb6206c0c26de46d9c50abae`。参数主要参考论文 v3 附录 B、Qwen3-1.7B 启动脚本，以及实际损失和数据 collator。以下记录用于下一轮配置设计，尚未修改或启动新的训练任务。

## 官方训练设置

| 参数 | 官方设置 | 上轮试跑 |
| --- | --- | --- |
| 模型 | Qwen3-1.7B | Qwen2.5-Math-1.5B-Instruct |
| 优化器 | AdamW | 学生 SGD，教师 AdamW |
| 学生学习率 | 5e-6 | 1e-4，且优化器不同 |
| 有效 batch | 32 | 1 |
| LoRA rank / alpha | 64 / 128 | 8 / 16 |
| LoRA 位置 | q、k、v、o、gate、up、down projections | q、v projections |
| 最多生成 | 1024 tokens | 128 tokens |
| 训练采样 | temperature 1.1，top_p 0.95，top_k 20 | temperature 0.7，top_p 0.95，未显式控制 top_k |
| 分布匹配温度 | 1.1，与采样共用 temperature | 1.0 |
| 损失 | full-vocabulary forward KL，pointwise clip 0.05 | 标准未裁剪 forward KL |
| 梯度裁剪 | max_grad_norm 0.1 | 未启用 |
| 教师 | 初始模型，固定 | 双 LoRA 元教师，动态更新 |
| 训练 thinking | 学生关闭，教师开启 | 没有 Qwen3 模式控制 |
| 论文报告预算 | 100 optimizer steps | 100 optimizer steps |
| 保存间隔 | 25 steps | 50 steps |

论文附录明确使用 AdamW 与 BF16。[论文 v3 附录 B](https://arxiv.org/html/2601.18734v3#A2)

官方 Qwen3-1.7B 脚本使用 4 个进程、每卡 batch 4、梯度累积 2，推得有效 batch 32。注意启动脚本设 num_train_epochs=30，并非硬编码 max_steps=100；论文和 README 对比的重点是 25、50、75、100 步 checkpoint。[固定版本启动脚本](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/scripts/run_opsd_1b.sh)

因此，同样写“100 步”，官方大约处理 3200 个学生题目 rollout，而上轮只有 100 个学生 rollout，外加 100 个 Quiz rollout。学生训练题次不匹配，不能直接比较方法优劣。Quiz 额外计算还需要单独纳入计算预算。

## 需要精确对齐的实现

官方 generalized_jsd_loss 在 beta=0 时调用 forward KL；这里 beta 是散度选项，不是额外 KL 正则的权重。它先对两侧 logits 除以 temperature，再算 log_softmax。没有额外乘 temperature 的平方；当前实现会乘这个因子，在 temperature=1 时没有差异，改为 1.1 时必须留意。

clip 发生在形状为 batch × sequence × vocabulary 的贡献张量上，对每个词表条目的 KL 贡献执行 clamp(max=0.05)，然后 mask 和归约；不是先对 vocabulary 求和再裁剪每个生成位置的总 KL，也不是 PPO 比率裁剪。代码没有将负贡献截成 0。该裁剪目标的数值不应被当作严格非负的完整 KL。

官方带 labels 的 batchmean 以有效生成位置总数为分母；当前单题序列平均在 batch=1 时一致，扩大 batch 后需明确是否按全部 token 加权。[固定版本损失实现](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py)

Qwen3 训练 collator 明确传入 student_thinking=False、teacher_thinking=True；学生问题提示和教师特权提示都使用各自的聊天模板。该模式是官方主实验的一部分，不能由 tokenizer 的默认行为代替。[固定版本 collator](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/data_collator.py)

官方训练入口使用 `siyanzhao/Openthoughts_math_30k_opsd`，并非当前 MATH 训练划分；当前数据集页面显示约 29.4k 条。若沿用 MATH，应称为在自建 MATH 划分上的 OPSD 实验，而非官方结果复现。[训练入口](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_train.py)、[训练数据](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)

## 评测设置

论文附录使用 thinking 开启、最大输出 38912、temperature=1.0、每题 12 个样本；这与上轮 50 道 MATH 开发题、greedy、1024-token 上限不同。论文表中 top_p=0.95，官方仓库 README 的评测文字存在另一种描述；需要复现时应固定实际 evaluate_math.py 的版本与调用参数，而非混用来源。

训练时关闭学生 thinking 与测试时开启 thinking 是官方选择，两者要作为独立配置字段保存。[论文评测参数](https://arxiv.org/html/2601.18734v3#A2)、[官方评测调用](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/eval/run_eval.sh)

## 对本项目的调整建议

先使用服务器现有的 `/home/yangdejin/OPSD/model/Qwen3-1.7B` 建立固定教师 OPSD 对照，优先对齐提示、thinking 模式、损失、优化器、采样与生成长度。单卡可先以 microbatch=1、梯度累积=32 保持基线有效 batch 32；这是待实测的工程配置，不保证完整长上下文能装入 24GB。

尽量保留官方 rank 64、alpha 128、七类投影。若元梯度分支显存不足，先测 rank 16、alpha 32 的适配版本，并将降低 rank 的变化同时应用于对照；不要一边用 rank 64 基线、一边用 rank 8 元学习后直接归因于方法。

max_completion_length=1024 只限制输出，不是教师总序列长。教师总长度还包含问题、参考解答、模板和生成前缀。官方脚本 max_length=20000 不应直接套给带高阶计算图的单张 3090；需要检查真实长度分布和显存，记录筛选覆盖率。

元学习教师的学习率、内层学习率、Quiz batch 和内层优化器是本项目新增超参数，OPSD 没有给出它们。若真实学生改为 AdamW，内层继续用 SGD 属于近似，需要说明并消融；不能把数值同为 5e-6 就当作两者更新等价。

基线的“梯度累积 32”是每次学生更新前积累 32 个样本，不等于把单题元学习更新跑 32 次；外层积累 32 个独立 teaching episode 也不等于一次 32 题内层模拟更新。正式比较应同时报告学生题次、Quiz 题次、optimizer steps、生成 tokens 和计算时间。

# 新服务器单卡 Dual OPSD 容量测试

测试服务器为 `222.19.197.47`，只使用 GPU 0（一张 RTX 3090，标称 24 GiB）。主机内存约 125 GiB，开始测试时可用约 121 GiB。GPU 1 未用于本轮测试。

## 测试条件

模型为服务器完整的 Qwen3-1.7B；BF16 骨干、FP32 双 LoRA，rank=64、alpha=128、七类投影。沿用 `configs/meta_qwen3.yaml` 的 JVP 元梯度、CPU 激活暂存、AdamW 和温度/裁剪设置。两个优化器的 moment 状态在测量前分配，模拟后续训练的常驻开销。

从当前长度过滤后的数据中分别选取最长学生和教师提示，用固定 1024-token 人工续写施加压力。蒸馏侧学生/教师总长度为 2047/4057，Quiz 侧为 1677/3029；Quiz batch 固定为配置中的 1。不同提示的最大值组合用于保守测量，不作为正确率实验。

探针完成临时学生更新、Quiz 反馈、教师更新和真实学生更新，检查梯度有限；不保存 checkpoint。记录 PyTorch allocated/reserved 峰值及进程 RSS 峰值。表中耗时不包含真实 on-policy rollout 生成，也不代表完整训练速度。

## Microbatch 扫描

Quiz 固定为 1 的单 episode 扩展路径记录位于 `outputs/dual_capacity_20261113_v2`：

| micro | allocated 峰值 | reserved 峰值 | 内存 RSS 峰值 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 1 | 10.02 GiB | 10.80 GiB | 21.26 GiB | 通过 |
| 2 | 15.24 GiB | 19.50 GiB | 40.64 GiB | 通过 |
| 3 | 20.45 GiB | 22.79 GiB | 60.05 GiB | 通过，显存余量较小 |
| 4 | — | — | — | CUDA OOM |

日常容量建议取 micro=2；这些上限仅针对上述长度、LoRA 和 offload 设置。

## 固定 micro=2 的大内层 batch 累积

权威累积记录位于 `outputs/dual_capacity_20261113_v4`。

| micro | 累积次数 | 有效 batch | allocated 峰值 | reserved 峰值 | 内存 RSS 峰值 | 不含 rollout 的耗时 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 | 16 | 15.78 GiB | 16.91 GiB | 40.68 GiB | 294.56 秒 | 通过 |
| 2 | 16 | 32 | 15.78 GiB | 16.91 GiB | 40.72 GiB | 562.30 秒 | 通过 |

累积的含义是先对所有 microbatch 的蒸馏梯度按有效 token 数加权求平均，再构造一次共享的临时学生 SGD 更新；Quiz 只评估这个临时学生。随后重新逐 microbatch 计算并累积其教师混合二阶导数，更新教师，最后对更新后的教师进行学生梯度累积与真实 AdamW 更新。

没有保留跨 microbatch 的完整反向图，也没有把多个独立 teaching episode 的元梯度平均冒充一个大内层 batch。`tests/test_accumulated_meta_probe.py` 已用 FP64 小模型将此累积超梯度与完整大 batch 展开逐参数比较，覆盖不等长响应、裁剪和不裁剪，并检查真实参数未被临时更新改变。

内存主要随 microbatch 和单条序列长度变化；累积时保留的是固定大小的梯度缓冲，次数增加不要求按同样倍数增加显存或主机内存。不要把已测到的最大累积次数解释为 RAM 的物理上限。

## 与正式训练代码的关系

容量验证完成后，按用户要求将共享内层累积路径接入正式 `Trainer`；`configs/meta_qwen3.yaml` 的原始 micro=1、累积=1 保留，新训练配置单独放在 `configs/meta_qwen3_b32_100.yaml`。共享内层梯度数值校验、20 项回归测试和 CUDA 微型模型的累积训练/精确恢复均已通过。

新配置使用单卡 micro=2、累积=16、有效 batch=32、Quiz batch=1，100 次更新。真实学生仍使用 AdamW，元反馈模拟为一次全有效 batch 的 SGD 更新；这是与原来单题内层相比的 batch 扩展。每次更新先在同一学生参数下采样所有训练轨迹，再进行共享内层模拟、教师更新和真实学生更新。梯度按有效输出 token 数加权，不按 microbatch 损失简单等权。

容量原始记录中关于正式 Trainer 仅支持 batch=1 的标记描述的是接入前的状态；实际训练以新配置、代码快照和运行 manifest 为准。

## 不等长 padding 验证

正式训练的同一 microbatch 会包含不同长度的序列。补充测试发现 PyTorch 2.8 的 efficient attention backward 对自定义 attention bias 有步长对齐要求；已在 `dual_opsd/models.py` 中将物理行宽对齐到 8，并在 CPU 激活暂存还原时保留该步长。

修复后的完整 Qwen3-1.7B 不等长测试通过，记录位于 `outputs/dual_capacity_20261113_v6_padded_aligned`：micro=2、累积=1、Quiz=1，allocated 峰值 15.51 GiB、reserved 峰值 17.03 GiB、进程 RSS 峰值 43.67 GiB。新增的 CUDA 回归测试也通过，修复后再次完成了微型模型的累积训练和 checkpoint 精确恢复测试。该测试与上面的累积容量测试共同构成正式运行的启动验证，不表示已测量正式训练全程峰值。

## 100 步正式运行

容量和 padding 验证完成后，已在服务器 GPU 0 启动全新训练，目录为 `/home/yangdejin/OPSD/outputs/meta_qwen3_b32_100_20261113_v1`，配置为 `configs/meta_qwen3_b32_100.yaml`。100 步指 100 次优化器更新，每步处理 32 条蒸馏样本和 1 条 Quiz 样本。第 1、25、50、75、100 步保存 checkpoint。

运行目录保留配置、代码快照、源文件 SHA256、manifest、进程号和 `training.log`；完整更新指标写入 `training/metrics.jsonl`。启动时已确认进入第 1 步真实 rollout，训练完成情况以这些服务器实时记录为准。容量表中的 9 分 22 秒不包含 rollout，不可直接用来推算 100 步总时长。

运行容量探针：

```bash
conda activate opsd
cd /home/yangdejin/OPSD
python scripts/profile_dual_capacity.py --config configs/meta_qwen3.yaml \
  --output outputs/dual_capacity_NEW --gpu 0 \
  --fixed-micro 2 --accumulations 8 16
```

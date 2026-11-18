# 本轮双路 OPSD 的固定教师对照

对照对象为 `outputs/meta_qwen3_b32_100_20261113_v1`。基线从初始 Qwen3-1.7B 新建学生 LoRA，不加载双路训练后的适配器。教师评分关闭全部 LoRA，使用冻结初始骨干；只更新学生，不采样 Quiz，也不计算教师元梯度。为兼容已有 checkpoint 格式，模型仍建立一套不参与更新的 teacher 适配器。

训练使用相同模型、种子、数据划分和长度资格限制：micro=2、累积16、有效 batch=32、100次学生 AdamW 更新，lr=5e-6、线性调度、梯度裁剪0.1；rank64/alpha128，学生 thinking 关闭、教师 thinking 开启，生成上限1024、温度1.1、top_p=0.95、top_k=20，逐词表项裁剪0.05。总长度4096、学生总长度2048，保持同一蒸馏样本池。

`configs/opsd_qwen3_b32_100.yaml` 是便于审阅的配置模板。正式入口从已完成双路实验的 `run_config.yaml` 生成冻结配置，避免未来改动模板影响对照。将双路日志中的100组、每组32个题目ID写入带 SHA256 的 `distill_schedule.json` 并重放。轨迹由基线学生重新 on-policy 生成，不复用双路回答；Quiz消耗随机数的差异和不同模型输出意味着生成轨迹不要求相同。普通基线启用 non-reentrant gradient checkpointing 以控制显存，其余共享训练参数保持一致。

这是学生题次、更新次数和生成上限匹配的对照。双路额外使用100次Quiz和元梯度计算，实际生成token数、运行时间和算力预算不相等。这一受控对照也不等同于官方全量数据、20k长度资格设置的完整复现。

## 准备与启动

服务器 `222.19.197.47` 的项目根目录：

```bash
cd /home/yangdejin/OPSD
source /home/yangdejin/miniconda3/etc/profile.d/conda.sh
conda activate opsd

# 只准备和校验配置、数据清单、3200条题目顺序；不会加载模型或开始训练。
python scripts/run_opsd_baseline.py \
  --reference-run outputs/meta_qwen3_b32_100_20261113_v1 \
  --output-root outputs/opsd_qwen3_b32_100_v1 \
  --prepare-only

# 训练100步，完成审计后自动进行AIME24/25评测和最终对照汇总。
# 此命令前台运行；长时间运行时请放入 tmux。
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_opsd_baseline.py \
  --reference-run outputs/meta_qwen3_b32_100_20261113_v1 \
  --output-root outputs/opsd_qwen3_b32_100_v1 \
  --evaluate
```

不带 `--evaluate` 只训练。入口不自动在后台启动，返回码会反映训练或评测失败。训练中断后，在相同命令增加 `--resume outputs/opsd_qwen3_b32_100_v1/training/checkpoints/step_000025`；恢复对应checkpoint内的优化器、scheduler、题目位置和随机状态。已有完整第100步checkpoint会跳过训练，重新进行审计；已落盘的评测回答会续跑。

第1、25、50、75、100步保存checkpoint。`training/metrics.jsonl` 每步记录全部32个ID、有效batch、loss、截断率、梯度范数和耗时。`training_audit.json` 检查配置、完整题序、长度过滤覆盖率和数据manifest相符。不要用旧 `opsd_qwen3_matched.yaml` 的 batch=1 或 `opsd_qwen3.yaml` 的20k长度设置替代本轮基线。

## 评测与结果

复用已验证的 vLLM 0.11.0 配对评测入口。AIME24/25各30题，每题12次，thinking开启，max_new_tokens=38912，temperature=1.0、top_p=0.95、top_k禁用、min_p=0、presence_penalty=0、max_model_len=40960，配对种子规则和题目顺序保持一致。基线训练前后共1440条回答，不运行MATH-500。

结果保存在 `outputs/opsd_qwen3_b32_100_v1/aime24_aime25_38912_s12`。总 `comparison.json` 报告本轮初始模型与固定教师OPSD的Avg@12；`opsd_vs_dual_comparison.json` 汇总固定教师和双路结果、截断率及准确率差异。单独汇总命令：

```bash
python scripts/compare_opsd_baseline.py \
  --meta-root outputs/meta_qwen3_b32_100_20261113_v1 \
  --opsd-root outputs/opsd_qwen3_b32_100_v1
```

汇总会拒绝数据、提示、随机种子规则、解码协议、版本或预算不一致，以及未完成的结果。两次初始模型评测结果分别保留；即使随机种子相同，GPU批调度和数值差异也可能造成回答差异，不把一轮初始结果覆盖到另一轮。本次以最终第100步为主比较，不用测试集挑选最佳checkpoint。

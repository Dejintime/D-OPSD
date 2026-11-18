# 首轮短程实验

配置：Qwen2.5-Math-1.5B-Instruct，共享冻结骨干，rank 8 的教师和学生 LoRA，100 次 meta OPSD 更新。训练总长上限 512，最多生成 128 token，每次内层与 Quiz 各 1 题，一步可微 SGD。损失为标准 forward KL。

训练数据沿用固定的 MATH 蒸馏与 Quiz 划分。评测从 500 道开发题中用独立随机种子 42 抽取 50 题，覆盖 7 个学科：代数 18、初等代数 7、中等代数 7、数论 6、几何 6、计数概率 4、预备微积分 2。训练前后固定同一批题，greedy 解码，上限 1024 个输出 token，使用 math-verify 判分。

开发集小样本用于初步观察，最终测试集未参与这一轮。50 题每变化 1 题，准确率变化 2 个百分点；100 步短程结果不能证明稳定的泛化收益。Quiz KL 的降低也不能替代真实答题正确率。

```bash
cd /home/yangdejin/OPSD
source .venv/bin/activate
python scripts/download_model.py --range-mirror
python scripts/run_pilot.py --config configs/server_debug.yaml \
  --steps 100 --eval-limit 50 --output outputs/pilot_20261111_v1
```

分段下载只支持固定的默认 checkpoint，合并后对完整权重做 SHA256 校验，匹配固定 Hugging Face revision 的权重 hash 后才能安装。

运行目录中的 status.json 标记当前阶段，baseline.log、training.log、trained.log 保存输出；comparison.json 保存配对前后准确率、改善题目与退步题目。训练中间结果和两套 LoRA 在 training/checkpoints 内。run_pilot.py 可在相同配置下再次运行以复用完成阶段，并从最新完整训练 checkpoint 恢复。

## 首轮结果

本轮已完成，固定 50 道开发题的配对结果如下：

| 指标 | 初始学生 | 100 步元训练后 |
| --- | ---: | ---: |
| 正确题数 | 12 / 50 | 15 / 50 |
| 正确率 | 24% | 30% |
| 答案提取失败率 | 0% | 2% |
| 1024-token 评测截断率 | 8% | 14% |

净提升 6 个百分点；4 题由错变对，1 题由对变错。变化题的最终答案已核对，正确率只评价最终答案，不代表整段推理均正确。训练后输出截断率反而升高，应在后续实验中检查长度和推理过程质量。

训练循环累计 441.99 秒，平均 4.42 秒/步；峰值 allocated 9.20 GiB，reserved 9.86 GiB。加上两次评测及加载，本轮流程总耗时约 21 分 18 秒，不包含此前权重下载。

100 步的平均 Quiz learning progress 为 0.0000519，51 步为正、49 步非正，尚未出现稳定的单步教学收益。训练 rollout 的截断率为 100%，意味着全部训练样本都达到 128 个输出 token 的上限。本轮是短前缀蒸馏试跑；观察到的 3 道净增益不是多种子统计结果，也不能归因于元反馈，因为尚未运行固定教师 OPSD 对照。

下一轮优先延长训练 rollout，并在相同数据、长度与学生训练预算下比较固定教师 OPSD 和 meta OPSD；随后扩大开发集评测与随机种子数量。最终 MATH-500 测试集未在本轮使用。

服务器结果目录为 `/home/yangdejin/OPSD/outputs/pilot_20261111_v1`。两套 LoRA 保存于 `training/checkpoints/step_000100/adapters/student` 和 `training/checkpoints/step_000100/adapters/teacher`。所有本轮训练、评测进程均已结束；服务器上其他用户的 GPU 任务未作操作。

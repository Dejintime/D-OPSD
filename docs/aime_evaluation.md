# AIME24 / AIME25 训练前后评测

当前仅评测 AIME24、AIME25。MATH-500 4k 单次采样评测已经结束，已有结果保留，不再启动或排队运行 MATH-500。

本轮训练目录：`/home/yangdejin/OPSD/outputs/meta_qwen3_b32_100_20261113_v1`，服务器为 `222.19.197.47`。本轮使用 micro=2、梯度累积=16、有效 batch=32 的第100步学生 checkpoint。

当前评测目录：`outputs/meta_qwen3_b32_100_20261113_v1/aime24_aime25_38912_s12`。旧 batch=1 实验的同名评测保留在 `outputs/meta_qwen3_100_20261111_v1` 下，不混用其结果。

## 数据与参数

使用官方 OPSD 评测脚本指定的来源，每套完整保留 30 道题：

| Benchmark | 数据源 | 固定 revision |
| --- | --- | --- |
| AIME24 | `HuggingFaceH4/aime_2024` | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` |
| AIME25 | `yentinglin/aime_2025` | `6f71d77b0b89b9dabe07ab466c51df33f514df7f` |

两套 Hugging Face 数据的 split 名均为 `train`，本项目将它们用于 benchmark 评测，不加入本轮训练。`scripts/prepare_aime.py` 固定源文件 hash、完整题数、跨年份题目去重检查，并保存官方行 ID 与独立 benchmark ID。处理后的文件和 manifest 位于 `data/processed/aime`。

采样参数对齐 [OPSD 论文 v3 表 8](https://arxiv.org/html/2601.18734v3#A2)：

| 参数 | 值 |
| --- | --- |
| thinking | 开启 |
| 每题采样数 | 12 |
| 最大输出 token 数 | 38912 |
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | -1，禁用 |
| min_p | 0.0 |
| presence_penalty | 0.0 |
| max_model_len | 40960，含输入与输出 |

评测只向学生输入题目，使用与官方一致的数学问答提示；取最后一个 boxed 答案，通过 math-verify 0.8.0 判分。初始 Qwen3-1.7B 为训练前模型，第 100 步学生 LoRA 为训练后模型，不加载教师 LoRA 推理。

每个 benchmark 每侧生成 360 条回答，两套前后合计 1440 条。主指标为 Avg@12，即 360 条回答的平均正确率；不是至少一次答对的 pass@12。

## 单卡运行与恢复

使用新服务器 `opsd` conda 环境中的 vLLM 0.11.0、torch 2.8.0、transformers 4.57.1。3090 单卡 GPU 0，tensor_parallel_size=1，max_num_seqs=8。硬件执行配置与官方多卡启动方式不同，采样预算保持与上一轮相同。绝对路径 Python 启动时设置 `TRITON_LIBCUDA_PATH=/home/yangdejin/miniconda3/envs/opsd/lib`。

同一引擎交错评测初始模型与训练后学生。每个题目及样本编号使用确定的配对种子，前后相同。结果逐条写入 JSONL；重启时恢复已有记录，metadata 不一致则拒绝混用。

`scripts/run_aime_comparison.py` 顺序启动 AIME24、AIME25，防止两个引擎同时争用显存。每阶段最多自动重试 3 次，支持从已完成年份继续；最终汇总分别报告两套 benchmark 的前后 Avg@12。

```bash
cd /home/yangdejin/OPSD
cat outputs/meta_qwen3_b32_100_20261113_v1/current_evaluation.json
cat outputs/meta_qwen3_b32_100_20261113_v1/aime24_aime25_38912_s12/suite_state.json
tail -n 20 outputs/meta_qwen3_b32_100_20261113_v1/aime24_aime25_38912_s12/aime24/vllm_pair.log
```

仅在原控制进程已停止时用下面的命令恢复：

```bash
conda activate opsd
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_aime_comparison.py \
  --root outputs/meta_qwen3_b32_100_20261113_v1 \
  --output-root outputs/meta_qwen3_b32_100_20261113_v1/aime24_aime25_38912_s12 \
  --max-num-seqs 8
```

每个年份的 `comparison.json` 保存该 benchmark 的前后结果；总目录中的 `comparison.json` 在两个年份均完成后生成。

# 新服务器环境与数据

服务器：`yangdejin@222.19.197.47`。项目目录：`/home/yangdejin/OPSD`。

## Conda 环境

环境名称为 `opsd`，路径为 `/home/yangdejin/miniconda3/envs/opsd`，使用 Python 3.12。依赖约束记录在 `requirements-opsd.txt`。

```bash
source /home/yangdejin/miniconda3/etc/profile.d/conda.sh
conda activate opsd
cd /home/yangdejin/OPSD
```

新环境将训练和 vLLM 评测合并为同一套依赖。vLLM 0.11.0 使用 PyTorch 2.8.0；Transformers 固定为 4.57.1、PEFT 为 0.19.1、math-verify 为 0.8.0。旧服务器训练曾使用另一套 PyTorch/Transformers，版本记录仍保留在 `requirements-lock.txt`，不能与当前约束混装。

该服务器的 NVIDIA 驱动只有版本化的 `libcuda.so.1`。已在 `opsd/lib/libcuda.so` 添加指向系统驱动的链接，并通过 conda 环境变量设置 `TRITON_LIBCUDA_PATH=/home/yangdejin/miniconda3/envs/opsd/lib`，供 vLLM 的 Triton 内核编译使用。运行时使用 `conda activate opsd` 或 `conda run -n opsd`，以加载该配置；没有修改系统驱动目录。

从空环境复建时：

```bash
conda create -n opsd python=3.12 pip -y
conda activate opsd
python -m pip install --no-cache-dir -r requirements-opsd.txt
python -m pip install --no-deps -e .
```

在同一服务器从空环境重建时，还需补上环境内链接和变量：

```bash
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /home/yangdejin/miniconda3/envs/opsd/lib/libcuda.so
conda env config vars set -n opsd TRITON_LIBCUDA_PATH=/home/yangdejin/miniconda3/envs/opsd/lib
conda deactivate
conda activate opsd
```

验证命令：

```bash
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 python scripts/smoke_test.py --device cuda --opsd-settings
OMP_NUM_THREADS=1 python scripts/check_environment.py --vllm-smoke
```

最后一项用临时生成的微型随机 Qwen3 权重检查 vLLM 的原始模型和 LoRA 推理，不下载正式模型，不产生 benchmark 准确率。验证记录和完整 pip freeze 位于 `outputs/setup_new_server`。

2026 年 11 月 13 日验收通过：pip 依赖检查无冲突，现有 19 项测试通过；CUDA 训练烟雾测试检查双适配器更新、冻结骨干、精确保存恢复、固定教师不变性；两张 RTX 3090 均通过 BF16 运算检查；vLLM 微型 Qwen3 原始模型和原生 LoRA 推理通过。完整依赖冻结文件为 `outputs/setup_new_server/pip_freeze.txt`，conda 导出为 `environment.yml`，总体验收记录为 `setup_complete.json`（后两项同在该输出目录）。

## 已准备的数据

沿用旧服务器已校验的固定数据快照，避免迁移后重新随机划分。所有 raw 与 processed 文件均按 manifest 核验 SHA-256。

| 数据 | 路径 | 数量 |
| --- | --- | ---: |
| 官方 OpenThoughts 源数据 | `data/raw/openthoughts` | 2 个 Parquet，共 29434 条 |
| 蒸馏训练集 | `data/processed/openthoughts/distill.jsonl` | 27925 |
| 教师元学习 Quiz | `data/processed/openthoughts/quiz.jsonl` | 1000 |
| 开发集 | `data/processed/openthoughts/dev.jsonl` | 500 |
| AIME24 | `data/processed/aime/aime24.jsonl` | 30 |
| AIME25 | `data/processed/aime/aime25.jsonl` | 30 |

OpenThoughts 数据源为 `siyanzhao/Openthoughts_math_30k_opsd`，revision `1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62`。AIME 两套基准的原始文件在 `data/raw/aime`，来源、revision 和 hash 见 `data/processed/aime/manifest.json`。

`openthoughts/test.jsonl` 是旧数据 manifest 中已有的 MATH-500 副本，作为数据完整性记录保留；当前仅安排 AIME24/25 作为数学 benchmark，没有启动新的 MATH-500 评测。

本次未下载正式模型权重。验收时检测到 `model/Qwen3-1.7B` 已有进行中的分片下载（`.incomplete`），需在下载结束后另行校验完整性；该下载任务未被本次操作干预。这次环境和数据准备没有启动训练或正式 benchmark 生成。

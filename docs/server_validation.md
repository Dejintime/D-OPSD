# 服务器验收记录

日期：2026 年 11 月 11 日。服务器项目目录：`/home/yangdejin/OPSD`；GPU：单张 RTX 3090，24GB。

## 已实现

共享冻结骨干与教师、学生双 PEFT LoRA；固定教师 OPSD、教师元优化和 SFT；学生 on-policy rollout；标准 forward KL；独立 Quiz 反馈；可微临时学生；真实学生 pilot 更新；数据预处理；评测；checkpoint 恢复；完整循环性能测量入口。

源代码在本地工作目录与服务器同步。服务器 `.venv` 复用 mopd 已有的核心组件，新增数学判分依赖仅安装到项目环境。核心版本为 PyTorch 2.10.0+cu128、Transformers 5.10.2、PEFT 0.19.1、datasets 4.3.0、math-verify 0.9.0。

## 已验证

- 7 项测试：KL 数值与方向、因果位置对齐、答案格式、分层划分、教师元梯度有限差分、零内层学习率反馈、参数分组与骨干冻结。
- 3090 上随机小 Qwen2 模型、BF16 骨干与 FP32 LoRA 的端到端检查：教师元梯度非零、两套适配器更新、骨干不变、checkpoint 恢复后下一步逐参数完全一致、固定教师不变、SFT 与评测入口可用。
- 完整 1.5B 架构的随机权重计算形状检查：rank 8、q/v LoRA、学生总长 256、教师总长 512、continuation 128；完成教师元梯度与学生梯度求解，峰值 allocated 8.305 GiB、reserved 8.404 GiB。
- 真实 tokenizer 与全量预处理数据的提示接口和长度筛选核对。

完整 1.5B 形状检查不包含自回归生成，教师手动更新用简单 SGD，未测 AdamW 稳态显存；它采用随机权重和固定 token，训练与 Quiz 复用相同视图。约 0.76 秒的计时只覆盖这段特定计算，不能作为完整训练循环或真实权重训练速度。随机模型的准确率没有研究意义。

测试与 GPU 小模型输出保存在服务器 `outputs/validation/`；完整形状检查和长度覆盖记录为 `outputs/fullsize_random_smoke.json`、`outputs/tokenizer_coverage.json`。

## 数据处理结果

固定 MATH revision 为 `21a5633873b6a120296cce3e2df9d5550074f4a3`，MATH-500 revision 为 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。

保留完整测试集，移除 2 道重复或重叠训练题；2 道未知难度训练题归入 level 0。实际数据数量如下，完整 ID 和 hash 见服务器 `data/processed/math/manifest.json`：

| 集合 | 题目数 |
| --- | ---: |
| 蒸馏 | 5998 |
| Quiz | 1000 |
| 开发 | 500 |
| MATH-500 测试 | 500 |

| 训练长度设置 | 保留蒸馏题 | 保留 Quiz |
| --- | ---: | ---: |
| 总长 512，预留生成 128 | 4075 | 681 |
| 总长 1024，预留生成 256 | 5587 | 930 |

以上是基于完整教师提示的长度筛选，不是模型准确率。正式实验须使用一致筛选规则并报告覆盖率；生成长度是否足够还需要真实模型截断率验证。

## 预训练权重与真实实验

预训练权重已在后续首轮实验中补全。权重共 3087467144 字节，SHA256 为 `08faf1e950173bb44c11d26df28c55da7237dc55c77dbc1cbae4a5d7584540d9`，与固定模型 revision 的 blob hash 一致。服务器本地模型目录现在可以加载。首轮 100 步元训练及 50 道固定开发题前后对比使用 outputs/pilot_20261111_v1，具体状态和结果见该目录的 status.json、comparison.json。

运行接口：

```bash
cd /home/yangdejin/OPSD
source .venv/bin/activate
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python scripts/download_model.py
python scripts/profile_step.py --config configs/server_debug.yaml \
  --steps 50 --warmup 3 --output outputs/profile_pretrained_001
python scripts/train.py --config configs/server_debug.yaml
```

download_model.py 在完整 snapshot 返回或并行权重 SHA256 校验通过且必要元信息存在后写入 download_manifest.json。首轮真实权重实验的最终结果需以 comparison.json 为准。独立 50 步性能测量与多随机种子主实验仍未运行。第一版标准 KL 与原稿未完整定义的 clipped KL 也应区分，不能混报。

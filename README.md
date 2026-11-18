# 双路 OPSD

默认模型为服务器上的 `model/Qwen3-1.7B`，默认数据为 `siyanzhao/Openthoughts_math_30k_opsd`。包含官方风格的固定教师 OPSD 基线，以及独立的双 LoRA 元学习实验分支。

## 环境

```bash
source /home/yangdejin/miniconda3/etc/profile.d/conda.sh
conda activate opsd
cd /home/yangdejin/OPSD
python -m pytest -q
```

## 数据

官方训练数据的两个 Parquet 分片已保存于 `data/raw/openthoughts/`，固定 revision 为 `1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62`，共 29434 条。

输出位于 `data/processed/openthoughts/`。为了支持研究中的独立 Quiz 和开发集，按题目划分为 27925 道蒸馏、1000 道 Quiz、500 道开发题。

### benchmark
当前数学 benchmark 为 `data/processed/aime/aime24.jsonl` 和 `aime25.jsonl`，各 30 题。

## 双 LoRA Meta Learning

当前主要的研究问题是：**如何优化教师的蒸馏分布，使学生经过一次学习后，在独立题目上的表现得到改善？**

模型共享冻结的 Qwen3-1.7B 骨干参数 $W_0$，并维护两套独立的 LoRA 参数：

- $\theta$：学生 LoRA，负责生成回答并接受蒸馏训练；
- $\phi$：教师 LoRA，负责根据参考解答提供蒸馏分布。

两者分别对应 $W_0+\Delta W_\theta$ 与 $W_0+\Delta W_\phi$，训练过程中不更新 $W_0$。

整体流程为：

> 学生生成轨迹 → 教师指导一次临时学习 → 独立 Quiz 评价临时学生 → 更新教师 → 使用更新后的教师训练真实学生。

### 1. 数据与条件分布

将训练数据按题目划分为蒸馏集 $\mathcal D_{\mathrm D}$ 和独立 Quiz 集 $\mathcal D_{\mathrm Q}$，两者不共享训练题目。每条数据记为：
$$
(x,z^\star),
$$
其中 $x$ 为题目，$z^\star$ 为数据集中提供的参考解答。

学生只接收题目，教师额外接收参考解答。令：$h_{\mathrm S}(x)$表示学生提示，$h_{\mathrm T}(x,z^\star)$ 表示包含参考解答的教师提示。在训练阶段中，学生关闭 thinking，教师开启 thinking。

定义用于损失计算的温度化分布：

$$
p_\theta^{(T)}(v\mid x,\hat y_{<t})
=
\operatorname{softmax}
\left(
\frac{a_\theta(h_{\mathrm S}(x),\hat y_{<t})}{T}
\right)_v,
$$

$$
q_\phi^{(T)}(v\mid x,z^\star,\hat y_{<t})
=
\operatorname{softmax}
\left(
\frac{b_\phi(h_{\mathrm T}(x,z^\star),\hat y_{<t})}{T}
\right)_v,
$$

其中 $v\in\mathcal V$，$\mathcal V$ 为完整词表，$T=1.1$。

另定义冻结参考分布：

$$
q_0^{(T)}(v\mid x,z^\star,\hat y_{<t}),
$$

其计算方式为关闭全部 LoRA，仅使用初始骨干 $W_0$ 和教师提示。它不依赖可学习教师参数 $\phi$。

学生回答由当前真实学生的采样策略生成：

$$
\hat y\sim\mu_\theta(\cdot\mid x).
$$
### 2. 内层目标：教师指导临时学生学习

从蒸馏集采样 batch $\mathcal B$，由更新前的学生 $\theta_k$ 为每道题生成一条回答。教师和学生在同一回答前缀上计算下一 token 分布，但接收不同的提示。

对每个回答位置和词表项，定义 forward KL 的逐项贡献：
$$
c_{i,t,v}(\theta,\phi)
=
q_\phi^{(T)}(v)
\left[
\log q_\phi^{(T)}(v)
-
\log p_\theta^{(T)}(v)
\right].
$$
省略的条件分别为该样本的学生提示、教师提示及相同回答前缀。

内层蒸馏目标为：
$$
\mathcal L_{\mathrm D}(\theta,\phi;\mathcal B)
=
\frac{1}{N_{\mathcal B}}
\sum_{(i,t)\in\mathcal I_{\mathcal B}}
\sum_{v\in\mathcal V}
\min\left(c_{i,t,v}(\theta,\phi),\tau\right),
$$
其中 $\tau=0.05$。

裁剪作用于求和前的每个词表项贡献，不是对每个位置的总 KL 裁剪。代码不额外乘 $T^2$，也不将负贡献截断为零，因此该裁剪目标允许出现负值，不能将其数值解释为严格非负的完整 KL。

随后构造一次临时 SGD 更新：

$$
\theta'_k(\phi)
=
\theta_k
-
\alpha
\nabla_\theta
\mathcal L_{\mathrm D}(\theta_k,\phi;\mathcal B),
$$

其中 $\alpha=10^{-4}$。

$\theta'_k$ 仅用于评价教师带来的学习效果，不覆盖真实学生参数 $\theta_k$，也不修改真实学生的 AdamW 状态。
### 3. Quiz 目标函数设计

Quiz 的作用是评价：**教师在蒸馏题目上提供的指导，是否能让临时学生在另一批独立题目上更接近可靠的参考分布？**

从独立 Quiz 集采样 $\mathcal Q$。Quiz 回答同样由更新前的真实学生生成：

$$
\hat y_j^{\mathrm Q}
\sim
\mu_{\theta_k}(\cdot\mid x_j^{\mathrm Q}).
$$

随后固定这些回答前缀，分别评价更新前学生 $\theta_k$ 和临时学生 $\theta'_k$。当前实现不会用临时学生重新采样 Quiz 回答。

令 $\mathcal I_{\mathcal Q}$ 为有效 Quiz 回答位置集合，$N_{\mathcal Q}=|\mathcal I_{\mathcal Q}|$，定义：

$$
\mathcal L_{\mathrm Q}(\vartheta;\mathcal Q)
=
\frac{1}{N_{\mathcal Q}}
\sum_{(j,t)\in\mathcal I_{\mathcal Q}}
D_{\mathrm{KL}}
\left(
q_0^{(T)}
\left(
\cdot\mid x_j^{\mathrm Q},z_j^{\star\mathrm Q},
\hat y_{j,<t}^{\mathrm Q}
\right)
\;\middle\|\;
p_\vartheta^{(T)}
\left(
\cdot\mid x_j^{\mathrm Q},
\hat y_{j,<t}^{\mathrm Q}
\right)
\right).
$$
展开为：
这一设计有三个关键点。

**第一，Quiz 使用冻结参考模型。**

Quiz 的目标分布来自 $q_0^{(T)}$，而非可学习教师 $q_\phi^{(T)}$。因此，教师不能通过直接改变 Quiz 的评分目标来降低损失；它只能通过影响临时学生的更新来影响 Quiz 结果。

**第二，Quiz 使用未裁剪的完整 forward KL。**

内层蒸馏保留 pointwise clipping，而外层 Quiz 不裁剪词表贡献，避免评分目标因裁剪而屏蔽部分分布差异。
在固定参考分布和回答前缀时：
$$
\mathcal L_{\mathrm Q}
=
H(q_0^{(T)},p_\vartheta^{(T)})
-
H(q_0^{(T)}),
$$
其中参考分布的熵对 $\vartheta$ 为常量。因此，最小化 Quiz KL 等价于最小化**冻结参考分布**与**学生分布**之间的软标签交叉熵。上式中的熵和交叉熵均按有效 Quiz 回答位置取平均。
### 4. 外层目标与教师元梯度

在第 $k$ 次更新中，将真实学生参数和采样轨迹视为固定，教师优化目标为：
$$
\min_\phi\;
\mathcal J_k(\phi)
=
\mathcal L_{\mathrm Q}
\left(
\theta'_k(\phi);\mathcal Q
\right).
$$
教师参数通过临时学生更新间接影响 Quiz：
$$
\phi
\longrightarrow
\nabla_\theta\mathcal L_{\mathrm D}
\longrightarrow
\theta'_k
\longrightarrow
\mathcal L_{\mathrm Q}.
$$
令：
$$
v_k
=
\left.
\nabla_\vartheta
\mathcal L_{\mathrm Q}(\vartheta;\mathcal Q)
\right|_{\vartheta=\theta'_k},
$$
并定义混合二阶导数矩阵：

$$
H_{\theta\phi}
=
\frac{\partial}{\partial\phi}
\left[
\nabla_\theta
\mathcal L_{\mathrm D}(\theta_k,\phi;\mathcal B)
\right].
$$

由链式法则：

$$
g_{\phi,k}
=
\nabla_\phi\mathcal J_k(\phi)
=
-\alpha H_{\theta\phi}^{\mathsf T}v_k.
$$

上式在当前教师参数 $\phi=\phi_k$ 处求值。这里没有 Quiz 损失关于教师参数的直接项，因为冻结参考分布与 $\phi$ 无关。

当前实现通过学生 logits 的 JVP 与教师侧反向传播计算该混合二阶导数和向量的乘积，不显式构造完整矩阵 $H_{\theta\phi}$。对于固定采样轨迹、一步 SGD 内层以及裁剪函数的分段可微区域，这对应同一计算图的链式法则。

教师使用该元梯度执行一次 AdamW 更新：

$$
\phi_{k+1}
=
\operatorname{AdamWStep}_{\eta_\phi}
\left(
\phi_k,\operatorname{ClipNorm}_{0.1}(g_{\phi,k})
\right),
$$

其中教师学习率 $\eta_\phi=10^{-5}$。AdamWStep 同时维护教师优化器的一阶、二阶动量状态，上式为省略这些状态的简记。

### 5. 真实学生的 Pilot 更新

更新教师后，丢弃临时学生参数 $\theta'_k$，重新从原始真实学生参数 $\theta_k$ 出发，使用更新后的教师评分同一批蒸馏轨迹：

$$
g_{\theta,k}
=
\left.
\nabla_\theta
\mathcal L_{\mathrm D}
\left(
\theta,\operatorname{sg}(\phi_{k+1});\mathcal B
\right)
\right|_{\theta=\theta_k},
$$

其中 $\operatorname{sg}$ 表示停止梯度。在这一步中，教师输出作为固定蒸馏目标。

真实学生执行：

$$
\theta_{k+1}
=
\operatorname{AdamWStep}_{\eta_{\theta,k}}
\left(
\theta_k,\operatorname{ClipNorm}_{0.1}(g_{\theta,k})
\right).
$$

学生初始学习率为 $5\times10^{-6}$，随后线性衰减。学生与教师分别维护独立的 AdamW 状态。
### 6. Batch 累积与学习收益指标

当前蒸馏有效 batch 为32，由 microbatch=2、累积16次实现。若第 $m$ 个 microbatch 包含 $N_m$ 个有效回答 token，则：

$$
\mathcal L_{\mathrm D}
=
\sum_{m=1}^{16}
\frac{N_m}{\sum_{\ell=1}^{16}N_\ell}
\mathcal L_{\mathrm D}^{(m)}.
$$

实现先累积全局 token 加权的学生梯度，再构造一次共享临时更新 $\theta'_k$，最后基于同一个 Quiz 反馈累积教师元梯度。它不是对16个独立临时学生的元梯度取平均。

当前 Quiz batch 为1。记录的学习收益为：

$$
\Delta_{\mathrm Q,k}
=
\mathcal L_{\mathrm Q}(\theta_k;\mathcal Q)
-
\mathcal L_{\mathrm Q}(\theta'_k;\mathcal Q).
$$

- $\Delta_{\mathrm Q,k}>0$：本次临时更新降低了该 Quiz 的参考分布匹配误差；
- $\Delta_{\mathrm Q,k}<0$：本次临时更新增加了该误差。

## 实验结果
**表1：Qwen3-1.7B在AIME24/25上的实验结果**

<table style="border-collapse:collapse; border-top:2px solid; border-bottom:2px solid;">
  <thead style="border-bottom:1px solid;">
    <tr>
      <th style="border:0; padding:8px;">方法</th>
      <th style="border:0; padding:8px;">AIME24 (%) ↑</th>
      <th style="border:0; padding:8px;">AIME25 (%) ↑</th>
      <th style="border:0; padding:8px;">平均 (%) ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:0; padding:8px;">初始Qwen3-1.7B</td>
      <td style="border:0; padding:8px; text-align:center;">51.94</td>
      <td style="border:0; padding:8px; text-align:center;">35.28</td>
      <td style="border:0; padding:8px; text-align:center;">43.61</td>
    </tr>
    <tr>
      <td style="border:0; padding:8px;">固定教师OPSD</td>
      <td style="border:0; padding:8px; text-align:center;">待完成</td>
      <td style="border:0; padding:8px; text-align:center;">待完成</td>
      <td style="border:0; padding:8px; text-align:center;">待完成</td>
    </tr>
    <tr>
      <td style="border:0; padding:8px;">双LoRA Meta OPSD</td>
      <td style="border:0; padding:8px; text-align:center;"><strong>53.06</strong></td>
      <td style="border:0; padding:8px; text-align:center;"><strong>38.33</strong></td>
      <td style="border:0; padding:8px; text-align:center;"><strong>45.69</strong></td>
    </tr>
  </tbody>
</table>

**注：** 指标为Avg@12，每套benchmark包含30道题，每题采样12次。
评测开启thinking，max_new_tokens=38912，temperature=1.0，
top_p=0.95，top_k禁用。

双LoRA实验训练100步，有效batch size为32；
相较初始模型，AIME24、AIME25和平均准确率分别提升
**1.11、3.06和2.08个百分点**。

固定教师OPSD采用相同学生训练预算，目前完成72/100步，
评测结果待补充。当前尚不能据此判断双路方法是否优于固定教师OPSD。

## 存在的问题和下一步的改进
### 1. 当前问题

**Quiz目标与答题能力可能存在偏差。** 当前Quiz通过KL衡量临时学生与冻结参考模型的分布差异。KL下降说明学生更接近参考分布，但不一定代表推理过程更正确或答题准确率提高。因此，教师收到的元反馈可能与实际能力提升不完全一致。

**Quiz反馈可能不够稳定。** 当前Quiz batch size为1，元梯度容易受到题目难度和采样轨迹影响，单题改善未必能够代表整体收益。

### 2. 下一步改进

计划使用独立Quiz题目的完整参考推导和答案，构造监督交叉熵：

$$
\mathcal L_{\mathrm Q}^{\mathrm{CE}}(\theta')
=
-\frac{1}{N_{\mathcal Q}}
\sum_{j,t}
\log p_{\theta'}
\left(
y_{j,t}^{\star}
\mid x_j,y_{j,<t}^{\star}
\right).
$$

其中，$\theta'$为接受教师指导后的临时学生，$y_j^\star$为标准解答，$N_{\mathcal Q}$为有效监督token总数。

这一修改的核心是：将**冻结模型的软分布监督**改为**标准解答token监督**，并在标准解答前缀上评价临时学生。Quiz损失仍用于更新教师，不直接作为真实学生的额外SFT更新。

同时，将Quiz batch size扩大到4或8，区分目标函数与反馈样本量的影响。
### 3. 后续方向

若上述改进有效，在学生到教师端，进一步引入PRM，识别学生推理中的错误步骤，为教师提供纠错与蒸馏权重调整信号，增强学生到教师的过程反馈。


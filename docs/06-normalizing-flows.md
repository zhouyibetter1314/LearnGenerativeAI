# 06 · 归一化流

[返回首页](../README.md) · [符号约定](notation.md)

> 将简单分布经过可逆变换变成复杂分布，同时用变量变换公式计算数据的密度。

## 先抓住这几件事

- Flow 同时提供生成方向 $z\to x$ 和推断方向 $x\to z$；两者由同一个双射连接。
- 密度随局部体积变化而变化，Jacobian 行列式是不能省略的修正项。
- 架构设计要兼顾表达能力、求逆速度和行列式计算成本；MAF 与 IAF 展示了两种不同取舍。

## 1. 为什么需要可逆生成模型？

一般潜变量模型需要边缘化：$p_{\theta}(x)=\int p_{\theta}(x\mid z)p(z)\,\mathrm dz$，积分通常难以精确计算。标准连续 Flow 改为确定性双射：

$$
z\sim p_Z(z),\qquad x=f_{\theta}(z),\qquad z=f_{\theta}^{-1}(x).
$$

本章统一约定：$f_{\theta}$ 是**生成方向**；$x,z\in\mathbb R^n$ 维数相同；变换及其逆满足变量变换所需的可微性，Jacobian 在相关区域非奇异。基分布 $p_Z$ 通常取标准高斯。

Flow 中每个数据点对应唯一潜变量，而非多个可能的潜在解释。其潜变量并不自动比数据低维，也不自动具有可解释语义。

## 2. 变量变换：保持概率质量，修正密度

### 2.1 一维例子

若 $Z\sim\mathcal U(0,2)$，$X=4Z$，则 $X\sim\mathcal U(0,8)$。相同概率质量被摊到四倍宽的区间，因此 $p_X(4)=1/8$，而不是 $p_Z(1)=1/2$。

对可逆、可微的一维变换：

$$
p_X(x)=p_Z(f^{-1}(x))\left|\frac{\mathrm d f^{-1}(x)}{\mathrm dx}\right|.
$$

绝对值兼容递增与递减变换。连续变量的单点概率为零，不能用 $P(X=x)=P(Z=z)$ 推出密度相等。

### 2.2 多维公式

记 $J_f(z)=\partial f(z)/\partial z$。局部体积满足 $\mathrm dx=|\det J_f(z)|\,\mathrm dz$，因此：

$$
\boxed{
p_{\theta}(x)=p_Z(f_{\theta}^{-1}(x))
\left|\det J_{f_{\theta}^{-1}}(x)\right|
=\frac{p_Z(z)}{|\det J_{f_{\theta}}(z)|}.
}
$$

相应的对数密度为：

$$
\log p_{\theta}(x)=\log p_Z(z)-\log|\det J_{f_{\theta}}(z)|.
$$

**检查方向的方法：**生成变换将体积放大时，密度应减小；若 $x=4z$，修正项就是 $-\log4$。

## 3. 多层组合与最大似然训练

令 $z_0=z$，$z_k=f_k(z_{k-1})$，$z_K=x$，则：

$$
f_{\theta}=f_K\circ\cdots\circ f_1,\qquad
\log p_{\theta}(x)=\log p_Z(z_0)
-\sum_{k=1}^{K}\log|\det J_{f_k}(z_{k-1})|.
$$

组合的 Jacobian 行列式相乘，所以对数行列式相加。由此得到训练、生成、推断三个流程：

| 任务 | 计算流程 |
| --- | --- |
| 训练 | 将数据 $x$ 逐层求逆得到 $z$，计算对数密度，最小化负对数似然 |
| 生成 | 从 $p_Z$ 采样 $z$，依次应用 $f_1,\ldots,f_K$ |
| 潜变量推断 | 直接计算 $f_{\theta}^{-1}(x)$ |

对数据集 $\{x^{(j)}\}_{j=1}^{N}$，目标为：

$$
\min_{\theta}\; -\frac1N\sum_{j=1}^{N}\log p_{\theta}(x^{(j)}).
$$

这里的“精确似然”表示无需变分下界或对潜变量积分；数值计算仍有浮点误差，而且精确可算不等于评估高维数据时成本很低。

## 4. 如何让 Jacobian 可计算？

稠密 $n\times n$ 矩阵的通用行列式计算通常需要 $O(n^3)$ 运算。若 Jacobian 为三角矩阵，则：

$$
\det J=\prod_{i=1}^{n}J_{ii},\qquad
\log|\det J|=\sum_{i=1}^{n}\log|J_{ii}|.
$$

这部分只需 $O(n)$ 运算；网络本身的前向计算并不因此自动变为 $O(n)$。

课件以 planar flow 说明“组合简单变换”的思路：

$$
f(z)=z+u\,h(w^{\top}z+b).
$$

它沿超平面相关方向扭曲分布。任意 $u,w,h$ 并不能自动保证可逆，实际需要合适的参数约束。后续架构更强调显式求逆与结构化 Jacobian。

### 4.1 NICE：加性耦合与缩放

将变量分成两组 $z=(z_a,z_b)$，定义：

$$
x_a=z_a,\qquad x_b=z_b+m_{\theta}(z_a).
$$

逆变换为 $z_a=x_a$，$z_b=x_b-m_{\theta}(x_a)$。其 Jacobian 为块三角矩阵，对角块均为单位阵，所以行列式为 $1$。

关键性质是：**子网络 $m_{\theta}$ 不必可逆，整个耦合层仍可逆。** 不变的一组变量提供了求逆所需的条件。多层需交换分组或置换坐标，使所有维度都能被变换。

加性耦合保持体积，NICE 因此加入正的逐维缩放 $x_i=s_i z_i$，使 $\log|\det J|=\sum_i\log s_i$。

### 4.2 Real NVP：仿射耦合

用条件缩放增加表达能力：

$$
x_a=z_a,\qquad
x_b=z_b\odot\exp(a_{\theta}(z_a))+m_{\theta}(z_a).
$$

逆变换与对数行列式为：

$$
z_b=(x_b-m_{\theta}(x_a))\odot\exp(-a_{\theta}(x_a)),\qquad
\log|\det J_f|=\sum_j a_{\theta,j}(z_a).
$$

指数保证缩放不为零。这里逆变换必须使用 **$\exp(-a)$**，不能与正向使用相同符号。

## 5. 自回归 Flow：MAF 与 IAF

以下均令 $z_i\overset{\mathrm{iid}}\sim\mathcal N(0,1)$，用 $a_i$ 表示**对数标准差**。因此，$x_i=\mu_i+e^{a_i}z_i$ 对应的方差是 $e^{2a_i}$。

### 5.1 MAF：条件参数依赖历史数据

$$
x_i=\mu_i(x_{<i})+\exp(a_i(x_{<i}))z_i,
\qquad
z_i=(x_i-\mu_i(x_{<i}))\exp(-a_i(x_{<i})).
$$

- **生成**：必须先生成 $x_{<i}$，才能算第 $i$ 个条件分布，因而按维度顺序执行。
- **评估给定数据的密度**：完整 $x$ 已知，可用掩码网络并行计算各维的 $\mu_i,a_i$，再并行求 $z_i$。

两方向的 Jacobian 均为三角结构，正向对数行列式为 $\sum_i a_i(x_{<i})$。这适合密度估计与最大似然训练。

### 5.2 IAF：条件参数依赖历史噪声

$$
x_i=\mu_i(z_{<i})+\exp(a_i(z_{<i}))z_i.
$$

- **生成**：先得到完整噪声向量 $z$，所有条件输入已知，因此可并行计算 $x_i$。
- **评估任意数据点的密度**：求逆时尚不知道 $z_{<i}$，必须逐维恢复潜变量。
- **评估自己刚生成的样本**：若保留了生成时的 $z$，可直接利用 Jacobian 计算密度，无需重新求逆。

| 架构 | 生成 $z\to x$ | 任意数据求逆 $x\to z$ | 典型用途 |
| --- | --- | --- | --- |
| NICE / Real NVP 耦合层 | 组内并行 | 组内并行 | 兼顾两方向效率 |
| MAF | 按维度顺序 | 可由掩码网络并行 | 密度估计 |
| IAF | 可由掩码网络并行 | 按维度顺序 | 快速生成、变分后验 |

“并行”描述每层内部的维度计算；多个 Flow 层仍需依次执行。IAF 的条件输入是 $z_{<i}$，不宜直接写成普通数据空间的 $p(x_i\mid x_{<i})$ 而混淆两种计算方向。

## 6. 概率密度蒸馏与 Parallel WaveNet

自回归音频模型 WaveNet 可以高效评估给定音频的密度，但逐采样点生成很慢。Parallel WaveNet 将两种优势组合起来：

1. 用最大似然训练自回归教师模型 $p_T$，发挥其密度评估优势。
2. 用 IAF 学生模型 $p_S$ 并行生成音频。
3. 固定教师，训练学生最小化：

$$
D_{\mathrm{KL}}(p_S\|p_T)
=\mathbb E_{x\sim p_S}[\log p_S(x)-\log p_T(x)].
$$

这个方向的 KL 只需学生采样、学生对自身样本的密度，以及教师对该样本的密度。学生保留生成噪声即可高效计算自身密度，教师也无需采样，因此避开了各自较慢的操作。

这是**概率分布蒸馏**，而非直接让学生逐点复制教师的一组输出。实际语音系统还可能加入任务相关辅助损失。

## 7. 能力与限制

- 标准连续 Flow 同时支持可计算密度、直接采样和唯一潜变量编码。
- 双射要求输入输出维数一致，对网络结构施加限制；它并非通常意义上的降维自编码器。
- 三角 Jacobian 降低行列式成本，但快速生成与快速求逆仍可能存在取舍。
- 对离散像素直接使用连续密度需要恰当的数据建模处理；连续密度值不等于某个离散图像的概率质量。
- 本讲正文覆盖离散层组合的 Flow；连续时间 Flow / Neural ODE 未展开，只在阅读清单中出现 FFJORD。

## 自检

1. **为什么变换可逆仍不足以高效计算密度？** 还需要高效求逆，并计算 Jacobian 的对数绝对行列式。
2. **Real NVP 的子网络也必须可逆吗？** 不必；保持不变的变量组使平移和缩放参数可直接重算。
3. **为什么 IAF 能快速评估自己的样本，却不能快速评估任意 $x$？** 前者已经知道生成噪声；后者需要顺序求逆恢复噪声。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文。页码均为 PDF 文件页码。

- [Lecture 7 · 动机与变量变换](../temp/Lecture7_Normalizing_Flow_I.pdf#page=8)，第 8–24 页；[Flow 构造、训练与计算复杂度](../temp/Lecture7_Normalizing_Flow_I.pdf#page=26)，第 26–37 页。
- [Lecture 8 · NICE 与 Real NVP](../temp/Lecture8_Normalizing_Flow_II.pdf#page=10)，第 10–25 页；[MAF / IAF](../temp/Lecture8_Normalizing_Flow_II.pdf#page=27)，第 27–43 页；[密度蒸馏与总结](../temp/Lecture8_Normalizing_Flow_II.pdf#page=45)，第 45–54 页。
- 公式核对：Lecture 8 第 21 页的 Real NVP 逆映射缩放符号应取负；本文统一高斯分布第二参数为方差，并将 $a_i$ 明确定义为对数标准差。

课件阅读清单包括 Rezende 与 Mohamed，*Variational Inference with Normalizing Flows*（2015）；Dinh 等，*NICE*（2014）与 *Density Estimation using Real NVP*（2017）；Kingma 等，*Improved Variational Inference with Inverse Autoregressive Flow*（2016）；Papamakarios 等，*Masked Autoregressive Flow for Density Estimation*（2017）。

进一步可读 *Glow*（2018）、*Invertible Residual Networks*（2019）、*FFJORD*（2019）、*Parallel WaveNet*（2018）、*ClariNet*（2019），以及教材 *Deep Generative Modeling* 第 4 章。

---

← [上一章：潜变量模型与 VAE](05-latent-variable-models-and-vae.md) · [目录](../README.md) · [下一章：生成对抗网络](07-generative-adversarial-networks.md) →

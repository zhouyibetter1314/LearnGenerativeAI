# 05 · 潜变量模型与变分自编码器

[返回首页](../README.md) · [符号约定](notation.md)

> 用潜变量解释数据的变化，用近似后验解决推断，再通过可微采样联合学习生成与推断网络。

## 先抓住这几件事

- **潜变量模型容易生成，却可能难以计算似然。** 给定 $z$ 后生成 $x$ 可以很简单，但求 $p_\theta(x)$ 要消去所有可能的 $z$。
- **ELBO 是对数似然的下界。** 下界与真实对数似然的差，恰好是近似后验到真实后验的 KL 散度。
- **VAE 的编码器输出分布参数，解码器定义条件分布。** 重参数化使重建项的梯度能够传回编码器；KL 项约束潜在表示与生成时的先验匹配。

## 1. 为什么引入看不见的变量？

人脸像素 $x$ 很复杂，但姿态、光照等潜在因素 $z$ 可能更紧凑。潜变量模型定义：

$$
p_\theta(x,z)=p(z)p_\theta(x\mid z),\qquad
p_\theta(x)=\int p(z)p_\theta(x\mid z)\,\mathrm{d}z.
$$

离散 $z$ 时把积分换为求和。三个分布承担不同任务：

| 分布 | 含义 | 用途 |
| --- | --- | --- |
| $p(z)$ | 潜变量先验 | 决定生成起点 |
| $p_\theta(x\mid z)$ | 给定潜变量的数据分布 | 解码、生成 |
| $p_\theta(z\mid x)$ | 观察数据后的真实后验 | 推断潜在因素 |

**高斯混合模型（GMM）** 是最简单的例子：

$$
P(Z=k)=\pi_k,\quad \sum_{k=1}^{K}\pi_k=1,\qquad
p_\theta(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x;\mu_k,\Sigma_k).
$$

生成时先按 $\pi$ 选择分量，再从对应高斯采样；聚类时计算分量后验：

$$
p_\theta(Z=k\mid x)=
\frac{\pi_k\mathcal{N}(x;\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal{N}(x;\mu_j,\Sigma_j)}.
$$

把离散分量索引换成连续 $z\sim\mathcal{N}(0,I)$，再用网络输出条件高斯的均值与协方差，就得到连续混合模型：简单的条件分布可以混合成复杂的边缘分布，参数量仍然有限。潜变量提供表征学习的途径，但**并不自动对应人类命名的语义因素**。

## 2. 困难在哪里：边缘化与后验

若联合分布已知，最大似然仍要优化 $\sum_{i=1}^{N}\log\int p_\theta(x^{(i)},z)\,\mathrm{d}z$。高维积分一般难算，真实后验也因归一化常数 $p_\theta(x)$ 难算而难以获得。

直接从先验采样估计 $p_\theta(x)$，可能很少采到能解释当前 $x$ 的潜变量。重要性采样改用更合适的提议分布 $q(z\mid x)$：

$$
p_\theta(x)
=\mathbb{E}_{q(z\mid x)}\left[\frac{p_\theta(x,z)}{q(z\mid x)}\right]
\approx\frac{1}{M}\sum_{m=1}^{M}
\frac{p_\theta(x,z^{(m)})}{q(z^{(m)}\mid x)},
\quad z^{(m)}\sim q(z\mid x).
$$

这里要求 $q$ 覆盖 $p_\theta(x,z)$ 的相关支撑集，且期望存在。好的 $q$ 应把概率质量放到能够解释 $x$ 的区域；否则权重可能高度不均、估计方差很大。

**似然估计无偏，不代表取对数后仍无偏。** 对正的重要性权重 $W$，Jensen 不等式给出 $\log\mathbb{E}[W]\geq\mathbb{E}[\log W]$，一般不相等；直接交换对数和期望会改变目标。这也引出了下界方法。

## 3. ELBO：将难算的对数似然变成可优化目标

令 $q_\phi(z\mid x)$ 为可计算的近似后验，定义证据下界（Evidence Lower Bound）：

$$
\mathcal{L}(x;\theta,\phi)
=\mathbb{E}_{q_\phi(z\mid x)}
\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\leq\log p_\theta(x).
$$

在相关 KL 与期望有限时，有精确恒等式：

$$
\log p_\theta(x)
=\mathcal{L}(x;\theta,\phi)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right).
$$

因此，**固定 $\theta$ 时，最大化 ELBO 等价于让近似后验靠近真实后验**；两者几乎处处相同时下界取等号。这是变分推断：用可优化的分布族近似难算的后验。

<details>
<summary>展开关键推导：Jensen 不等式与下界的差</summary>

对数是凹函数，故在重要性采样恒等式成立时：

$$
\log p_\theta(x)
=\log\mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\geq\mathbb{E}_{q_\phi}\!\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

再使用 $p_\theta(x,z)=p_\theta(z\mid x)p_\theta(x)$，从右侧分离出 $\log p_\theta(x)$，剩余项就是负的后验 KL。这里后验 KL 用来解释下界的差，**训练时无需直接计算这个难算的后验 KL**。

EM 与此有联系：精确 E 步在固定模型时令 $q$ 等于真实后验，随后 M 步优化模型参数；当真实后验不可计算时，可以改用受限分布族进行变分近似。课件只简要指出了这一联系。

</details>

将联合分布进一步分解，得到最常用的 VAE 形式：

$$
\boxed{\mathcal{L}(x;\theta,\phi)
=\underbrace{\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]}_{\text{重建对数似然}}
-\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))}_{\text{先验正则}}.}
$$

注意两个 KL 的对象不同：**后验 KL 衡量下界有多紧，先验 KL 是 ELBO 中直接计算的项**。联合优化 $\theta,\phi$ 可以提高下界，但下界提高不意味着真实对数似然在每一步都必然提高，因为下界的差也在变化。

## 4. VAE：用两个网络实现变分学习

对每个样本单独优化一个近似后验代价很高。**摊销推断（Amortized Inference）** 用共享编码器，根据输入直接输出近似后验的参数：

$$
q_\phi(z\mid x)=\mathcal{N}\!\left(z;\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\right),\qquad
p(z)=\mathcal{N}(0,I).
$$

编码器输出均值与对数方差，并非直接输出一个确定的 $z$。解码器则可定义：

$$
p_\theta(x\mid z)=\mathcal{N}\!\left(x;\mu_\theta(z),\sigma_0^2I\right).
$$

这里 $\sigma_0^2>0$ 是固定观测噪声方差，$n$ 是数据维数。于是：

$$
-\log p_\theta(x\mid z)
=\frac{\|x-\mu_\theta(z)\|_2^2}{2\sigma_0^2}
+\frac{n}{2}\log(2\pi\sigma_0^2).
$$

**只有在固定各向同性高斯观测模型下，负重建对数似然才等价于带固定系数的平方误差。** 若方差也需要学习，就不能丢弃其对数项；若观测是伯努利变量，则对应伯努利负对数似然。

对于上述对角高斯后验与标准高斯先验，KL 有闭式解。令潜变量维数为 $d_z$：

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
=\frac12\sum_{j=1}^{d_z}
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

因此，最小化的训练损失是**负 ELBO = 期望重建负对数似然 + 先验 KL**。重建项通常通过少量 Monte Carlo 样本估计，KL 则可精确计算。

## 5. 重参数化：如何穿过采样求梯度？

直接把“从依赖 $\phi$ 的分布中抽样”当作普通确定性操作，无法直接进行通常的反向传播。对于对角高斯，可以等价地写为：

$$
\epsilon\sim\mathcal{N}(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
$$

随机性来自与 $\phi$ 无关的 $\epsilon$；固定一次噪声后，$z$ 是关于编码器参数的可微函数，因此可以估计重建项的路径梯度。交换微分与期望需要相应可微性及可积性条件；该做法不能直接照搬到任意离散采样。

**一次训练迭代：**

1. 取小批量数据，用编码器计算 $\mu_\phi(x)$ 和 $\log\sigma_\phi^2(x)$。
2. 采样标准高斯噪声，经重参数化得到 $z$。
3. 解码并计算重建负对数似然，再加上先验 KL。
4. 对负 ELBO 反向传播，同时更新 $\theta,\phi$。

**生成新样本：** 从先验采样 $z\sim p(z)$，再采样 $x\sim p_\theta(x\mid z)$，无需编码器。只输出解码均值 $\mu_\theta(z)$ 是常见展示方式，但与采样完整观测分布不同。

## 6. 为什么重建好还不够？

普通自编码器（AE）可以把训练样本映射到一些孤立的潜在点，再准确重建；但它没有自动提供一个适合采样的潜变量分布。从任意位置采样，可能落到解码器没有学过的区域。

VAE 训练时，解码器实际接收的潜变量分布是**聚合后验**：

$$
q_\phi(z)=\int q_\phi(z\mid x)p_{\mathrm{data}}(x)\,\mathrm{d}x,
$$

而生成时接收的是 $p(z)$。逐样本的 KL 约束有助于缩小这种失配，但不能保证潜在空间没有“空洞”，也不能保证所有维度都具有可解释语义。

两项损失需要共同发挥作用：仅保留重建项，潜在表示可能不适合先验采样；仅最小化 KL，编码器可能不再保存关于输入的有效信息。

## 7. 重要变体与局限

### β-VAE：控制潜在信息容量

$$
\mathcal{L}_{\beta}(x)
=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\beta D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z)).
$$

$\beta=1$ 恢复标准 ELBO；常用 $\beta>1$ 加强信息瓶颈和因子化先验的约束，可能改善解耦表征，但也可能牺牲重建。独立高斯先验**不能单独保证语义解耦**；改变权重后，目标一般不再是标准 ELBO。

### 后验坍塌：模型不再使用潜变量

典型表现为 $q_\phi(z\mid x)\approx p(z)$、先验 KL 很小，且解码器几乎忽略 $z$。课件指出两种可能情形：解码器尚未学会利用潜变量时，KL 约束过早占主导；或者强自回归解码器仅靠已知历史就能预测数据，从而绕过 $z$。**低 KL 并不自动说明模型更好**，需结合重建与潜变量使用情况判断。

### VQ-VAE：用码本学习离散表示

令编码器输出 $z_e(x)$，码本为 $\{e_1,\ldots,e_K\}$，选择最近向量：

$$
k=\arg\min_j\|z_e(x)-e_j\|_2^2,\qquad z_q(x)=e_k.
$$

训练损失由重建项、码本项和承诺项组成：

$$
\mathcal{J}_{\mathrm{VQ}}
=-\log p_\theta(x\mid e_k)
+\|\operatorname{sg}[z_e(x)]-e_k\|_2^2
+\beta_{\mathrm{c}}\|z_e(x)-\operatorname{sg}[e_k]\|_2^2.
$$

$\operatorname{sg}$ 表示停止梯度；码本项更新嵌入，承诺项约束编码器靠近所选嵌入。最近邻选择不可微，通常通过直通梯度估计把重建梯度传给编码器。这里的 $\beta_{\mathrm{c}}$ 是承诺项权重，与 β-VAE 的 KL 权重含义不同。

在确定性离散编码、固定均匀先验的设定下，潜变量 KL 为常数，因此训练式不必包含一个可优化的高斯 KL 项。为生成高质量样本，通常还会另外学习离散索引的联合先验，例如使用 PixelCNN；**学习码本本身不等于学会了索引的联合分布**。

VQ-VAE-2 使用多层离散潜变量，分别捕捉全局结构与局部细节，并学习相应的条件先验。它也解释了为什么[自回归模型](03-autoregressive-models.md)可以从像素空间转移到潜在 token 空间。

### 后续改进方向

课件还指出：改进摊销推断的优化、使用更灵活的变分后验或可学习先验，以及采用 MMD、Wasserstein 距离等替代准则。IWAE 和逆自回归流（IAF）出现在阅读清单中，分别关联更紧的加权下界和更灵活的近似后验，本讲未展开其推导。

## 自检

1. **VAE 能直接计算精确的 $p_\theta(x)$ 吗？** 一般不能；它能计算或估计 ELBO，并通过重要性采样等方法估计似然。
2. **最大化 ELBO 等于最小化哪个 KL？** 固定模型时，等价于最小化近似后验到真实后验的 KL；实现中直接出现的则是近似后验到先验的 KL，二者不能混淆。
3. **生成时需要编码器吗？** 不需要。编码器服务于推断与训练；生成从先验开始，经解码器得到数据。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文；下面链接在本地仓库中可打开。

| 来源 | 对应内容 |
| --- | --- |
| [第 5 讲，第 12–29 页](../temp/Lecture5_Latent_Variable_Models_I_updata260924.pdf#page=12) | 潜变量、GMM、连续混合模型 |
| [第 5 讲，第 31–61 页](../temp/Lecture5_Latent_Variable_Models_I_updata260924.pdf#page=31) | 边缘似然、重要性采样、ELBO、变分推断 |
| [第 6 讲，第 4–33 页](../temp/Lecture6_Variational_Autoencoder_II_update260929.pdf#page=4) | 摊销推断、VAE 目标、重参数化 |
| [第 6 讲，第 35–52 页](../temp/Lecture6_Variational_Autoencoder_II_update260929.pdf#page=35) | AE 与 VAE 的潜在空间、聚合后验 |
| [第 6 讲，第 54–76 页](../temp/Lecture6_Variational_Autoencoder_II_update260929.pdf#page=54) | β-VAE、后验坍塌、VQ-VAE / VQ-VAE-2 |

课件推荐阅读：Kingma 与 Welling，*Auto-Encoding Variational Bayes*（ICLR 2014）；Rezende 等，*Stochastic Backpropagation and Approximate Inference in Deep Generative Models*（ICML 2014）；Burda 等，*Importance Weighted Autoencoders*（ICLR 2016）；Kingma 等，*Improved Variational Inference with Inverse Autoregressive Flow*（NeurIPS 2016）；Higgins 等，*β-VAE*（ICLR 2017）；Bowman 等，*Generating Sentences from a Continuous Space*（CoNLL 2016）；van den Oord 等，*Neural Discrete Representation Learning*（NeurIPS 2017）；He 等，*Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*（ICLR 2019）。

教材补充：*Deep Generative Modeling* 第 5 章；*The Principles of Diffusion Models*，B.2 节。VQ-VAE-2 示例来源为 Razavi 等，*Generating Diverse High-Fidelity Images with VQ-VAE-2*（NeurIPS 2019）。

---

← [上一章：最大似然与泛化](04-maximum-likelihood.md) · [目录](../README.md) · [下一章：归一化流](06-normalizing-flows.md) →

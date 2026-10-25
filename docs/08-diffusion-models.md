# 08 · 扩散模型：把生成拆成逐步去噪

[返回首页](../README.md) · [符号约定](notation.md)

> 预先定义加噪过程，用可计算的去噪目标学习反向生成过程。

## 先抓住这几件事

- 前向过程 $q$ 逐步破坏数据，反向模型 $p_\theta$ 学习恢复数据；两者的条件方向不同。
- 训练时知道干净样本 $x_0$，因此能计算 $q(x_{t-1}\mid x_t,x_0)$，为反向模型提供监督。
- DDPM 将变分训练转化为噪声预测；常用的无权重噪声 MSE 是重新加权后的目标，不能直接称为精确最大似然。

## 1. 前向加噪：一步就能构造任意噪声级别

令 $x_0\sim p_{\mathrm{data}}$，选定噪声日程 $0<\beta_t<1$，定义：

$$
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,\qquad \bar\alpha_0=1.
$$

前向马尔可夫过程为：

$$
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),\qquad
q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I).
$$

高斯噪声逐步合成后，有闭式边缘分布：

$$
q(x_t\mid x_0)=\mathcal{N}(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad \epsilon\sim\mathcal{N}(0,I).
$$

所以训练时不必真的从 $x_0$ 连续加噪 $t$ 次。注意这里的 $\epsilon$ 是合成后的总噪声，不是某一步独有的噪声。

当 $\bar\alpha_T$ 足够接近 $0$ 时，$q(x_T\mid x_0)$ 接近标准高斯。有限日程一般只是近似，并非自动精确等于 $\mathcal{N}(0,I)$。

## 2. 反向生成：从噪声到数据

模型定义为：

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),\qquad
p(x_T)=\mathcal{N}(0,I).
$$

常见反向参数化使用高斯分布：

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I).
$$

这里先讨论固定方差；方差也可以学习。有限步长下，真实的 $q(x_{t-1}\mid x_t)$ 未必是高斯，使用高斯反向转移属于模型选择，小步加噪为这种近似提供了依据。

```mermaid
flowchart LR
    A["干净数据 x₀"] -->|"固定加噪 q"| B["含噪数据 xₜ"]
    B -->|"继续加噪"| C["近似高斯噪声 xT"]
    C -->|"学习的反向转移"| D["逐步恢复结构"]
    D --> E["新样本"]
```

## 3. 为什么训练时可以得到去噪监督？

需要区分三个不同的条件分布：

| 分布 | 是否已知？ | 作用 |
| --- | --- | --- |
| $q(x_t\mid x_{t-1})$ | 已知 | 人为规定的前向加噪 |
| $q(x_{t-1}\mid x_t)$ | 通常难以直接计算 | 理想的反向转移，依赖真实数据分布 |
| $q(x_{t-1}\mid x_t,x_0)$ | 给定训练样本后可计算 | 训练时的解析监督目标 |

由贝叶斯公式与马尔可夫性，$t\geq2$ 时：

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
=\mathcal{N}(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I),
$$

其中参数为：

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad
\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

训练可以使用 $x_0$ 构造目标，但网络的输入只有 $x_t,t$，不会在生成时访问真实干净图像。它需要学习不同训练样本共同决定的去噪规律。

## 4. 训练目标：一个多层潜变量模型的 ELBO

把 $x_{1:T}$ 看作潜变量，固定的 $q(x_{1:T}\mid x_0)$ 充当变分分布。负 ELBO 是负对数似然的上界：

$$
-\log p_\theta(x_0)\leq\mathcal{J}_{\mathrm{VLB}}(x_0)
=\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}\right].
$$

其可计算分解为：

$$
\begin{aligned}
\mathcal{J}_{\mathrm{VLB}}(x_0)
={}&D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\Vert p(x_T)\bigr)\\
&+\sum_{t=2}^{T}\mathbb{E}_{q(x_t\mid x_0)}
D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\Vert p_\theta(x_{t-1}\mid x_t)\bigr)\\
&-\mathbb{E}_{q(x_1\mid x_0)}\log p_\theta(x_0\mid x_1).
\end{aligned}
$$

| 组成 | 含义 | 学习时要注意什么？ |
| --- | --- | --- |
| 终点先验匹配 | 加噪终点接近标准高斯 | 固定日程与先验时不依赖 $\theta$，通常很小但未必为零 |
| 中间转移匹配 | 反向模型逼近前向后验 | $t\geq2$ 的主要去噪训练项 |
| 重建项 | 从 $x_1$ 恢复 $x_0$ | $t=1$ 单独处理，不使用零方差高斯密度 |

<details>
<summary>展开：前向条件如何变成反向 KL？</summary>

由贝叶斯公式：

$$
q(x_t\mid x_{t-1})
=q(x_{t-1}\mid x_t,x_0)
\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.
$$

将其代入负 ELBO，并把 $t=1$ 单独取出。中间边缘项逐项消去：

$$
\prod_{t=2}^{T}\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}
=\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}.
$$

随后对前向轨迹取期望，就得到先验 KL、中间转移 KL 与重建负对数似然。与 VAE 相同，下界和真实对数似然之间还存在后验 KL 间隙；优化下界不能直接写成精确计算似然。

</details>

## 5. 从预测均值到预测噪声

把 $x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$ 代回 $\tilde\mu_t$，得到：

$$
\tilde\mu_t=
\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).
$$

因此用网络预测噪声，就能确定反向均值：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

固定 $\sigma_t^2>0$ 时，中间高斯 KL 中依赖均值的部分等价于加权噪声 MSE：

$$
w_t\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2,\qquad
w_t=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)},\quad t\geq2.
$$

课件取 $\sigma_t^2=\tilde\beta_t$ 讨论该关系。DDPM 常用的简化目标进一步去掉时间权重：

$$
\mathcal{L}_{\mathrm{simple}}
=\mathbb{E}_{x_0,\,t,\,\epsilon}
\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,t)\|_2^2\right],
\qquad t\sim\operatorname{Unif}\{1,\ldots,T\}.
$$

这是目标的重新加权，会改变不同噪声水平的相对贡献。包含 $t=1$ 的简化训练也不能与上面仅对 $t\geq2$ 推导的 KL 权重公式混为一谈。

## 6. 训练与生成怎么进行？

**训练一次更新：**

1. 从训练集选取 $x_0$，随机选取时间步 $t$。
2. 采样 $\epsilon\sim\mathcal{N}(0,I)$，直接构造 $x_t$。
3. 计算 $\epsilon_\theta(x_t,t)$，最小化噪声预测误差。

同一个网络通过时间条件 $t$ 处理所有噪声级别；可采用 U-Net 或 Transformer 等架构。一次更新只需随机时间步，**不必反向传播穿过整条采样链**。

**生成一个样本：** 从 $x_T\sim\mathcal{N}(0,I)$ 出发，对 $t=T,\ldots,1$ 迭代：

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
+\sigma_t z,\qquad z\sim\mathcal{N}(0,I).
$$

常见 DDPM 实现最后一步令 $z=0$，返回预测均值。这个输出约定不意味着可以把训练中的重建似然定义为零方差高斯密度。

在任意时间步，还可以给出干净数据的当前估计：

$$
\hat x_0(x_t,t)=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

它是模型估计，不是已知真实答案；可以观察估计随去噪逐步改善，也可在中间噪声状态进行插值后继续生成。前向噪声调度决定不同时间步还保留多少信息。

## 7. 与 VAE 的联系，以及方法的代价

扩散模型可以看作具有许多潜变量层的 VAE：生成方向逐层恢复数据，训练仍使用变分下界。区别是基本 DDPM 的前向 $q$ 由噪声日程固定，而常规 VAE 的近似后验 $q_\phi(z\mid x)$ 由编码器学习。

逐步解决局部去噪问题，带来了稳定的回归目标和较好的样本质量，也使采样需要多次调用网络。对离散数据的处理并不能直接照搬这里的连续高斯公式。

本课件只展开离散时间 DDPM。Score matching、SDE/ODE、DDIM、条件引导与潜在扩散出现在路线或阅读清单中，尚未给出完整推导；[Normalizing Flow](06-normalizing-flows.md) 的可逆密度变换也不能直接等同于这些后续主题。

## 自检

1. **为什么训练无需逐步加噪？** $q(x_t\mid x_0)$ 有闭式表达，可直接采样。
2. **网络训练时是否把 $x_0$ 当输入？** 这里的网络输入是 $x_t,t$；$x_0$ 用于构造训练噪声与监督。
3. **无权重噪声 MSE 是否就是精确负对数似然？** 不是，它来自变分目标的重新加权与简化。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文。以下页码为 PDF 页序。

- [第 11 讲：动机与正反过程，10–23 页](../temp/Lecture11_Diffusion_Models.pdf#page=10)；[ELBO 与目标分解，25–41 页](../temp/Lecture11_Diffusion_Models.pdf#page=25)。
- [前向后验、噪声参数化与算法，43–66 页](../temp/Lecture11_Diffusion_Models.pdf#page=43)；[层次 VAE、网络与讨论，69–82 页](../temp/Lecture11_Diffusion_Models.pdf#page=69)。
- 关键论文：Sohl-Dickstein 等，*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（2015）；Ho 等，*Denoising Diffusion Probabilistic Models*（2020）。
- 课件 83–84 页列出 DDIM、Score SDE、改进 DDPM、引导、Latent Diffusion 与 EDM 等延伸阅读；[完整来源索引](sources.md)。

---

← [上一章：生成对抗网络](07-generative-adversarial-networks.md) · [目录](../README.md)

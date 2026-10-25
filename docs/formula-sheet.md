# 核心公式速查

[返回首页](../README.md) · [符号约定](notation.md)

用于复习与定位，定义和推导请沿各节链接阅读。以下记号沿用符号页；不能省略旁边的适用条件。

## 1. 概率分解与贝叶斯公式

$$
p(x)=\prod_{i=1}^{n}p(x_i\mid x_{<i}),\qquad
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

链式分解不要求独立；贝叶斯公式要求分母非零。贝叶斯网络用局部父节点集合替代完整历史。→ [概率建模](02-probabilistic-modeling.md)

## 2. 最大似然与正向 KL

$$
\arg\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta)
=\arg\max_\theta\mathbb{E}_{p_{\mathrm{data}}}\log p_\theta(x),\qquad
\widehat{\mathcal{L}}_{\mathrm{NLL}}
=-\frac1N\sum_{k=1}^{N}\log p_\theta(x^{(k)}).
$$

总体等价关系需相关量有定义；右侧有限样本目标是总体期望的估计。→ [最大似然](04-maximum-likelihood.md)

## 3. 自回归似然

$$
\log p_\theta(x)=\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i}).
$$

训练使用真实历史，生成使用已采样历史。可并行训练不等于可以并行生成。→ [自回归模型](03-autoregressive-models.md)

## 4. ELBO 与后验间隙

$$
\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x)
&=\mathbb{E}_{q_\phi(z\mid x)}\log p_\theta(x\mid z)
-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)),\\
\log p_\theta(x)-\mathcal{L}_{\mathrm{ELBO}}(x)
&=D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)).
\end{aligned}
$$

先验正则 KL 与后验间隙 KL 是两个不同的量；训练最大化 ELBO。→ [潜变量与 VAE](05-latent-variable-models-and-vae.md)

## 5. 高斯 VAE 的实现

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I),
$$

$$
D_{\mathrm{KL}}\left(\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))\Vert\mathcal{N}(0,I)\right)
=\frac12\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2).
$$

固定各向同性高斯观测分布时，重建 NLL 才等价于带固定系数的 MSE。→ [高斯实现与重参数化](05-latent-variable-models-and-vae.md)

## 6. Flow 的密度变换

$$
x=f_\theta(z),\qquad
\log p_\theta(x)=\log p_Z(f_\theta^{-1}(x))
-\log\left|\det J_{f_\theta}(z)\right|.
$$

假设是同维可微双射且 Jacobian 非奇异。生成方向体积放大，密度相应减小。→ [归一化流](06-normalizing-flows.md)

## 7. GAN 与非饱和目标

$$
\min_G\max_D\left[\mathbb{E}_{p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_Z}\log(1-D(G(z)))\right],
\qquad
L_G^{\mathrm{NS}}=-\mathbb{E}_{z\sim p_Z}\log D(G(z)).
$$

原始 minimax 在理想最优判别器下对应 $-\log4+2\operatorname{JS}(p_{\mathrm{data}},p_\theta)$；非饱和目标改变了生成器的训练梯度。→ [GAN](07-generative-adversarial-networks.md)

## 8. Wasserstein 对偶

$$
W_1(p,q)=\sup_{\|f\|_{\mathrm{Lip}}\leq1}
\left(\mathbb{E}_{p}f(X)-\mathbb{E}_{q}f(X)\right).
$$

欧氏空间下需有限一阶矩；critic 是实值函数，梯度惩罚仅在采样点施加软约束。→ [WGAN](07-generative-adversarial-networks.md)

## 9. 扩散：前向加噪与噪声预测

$$
\alpha_t=1-\beta_t,\qquad \bar\alpha_t=\prod_{s=1}^{t}\alpha_s,\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right),\qquad
\mathcal{L}_{\mathrm{simple}}
=\mathbb{E}_{x_0,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2.
$$

这里 $\epsilon\sim\mathcal{N}(0,I)$、$t$ 均匀抽样；无权重目标是变分目标的简化，$t=1$ 的重建项需区分处理。→ [扩散模型](08-diffusion-models.md)

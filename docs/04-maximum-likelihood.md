# 04 · 最大似然与泛化

[返回首页](../README.md) · [符号约定](notation.md)

> 给观测数据更高的模型概率，并用独立数据检查这种拟合能否泛化。

## 先抓住这几件事

- **最大似然是一个学习准则**，可以用于不同模型族；它不限定网络架构。
- **总体目标与样本目标不同**：最小化总体正向 KL 等价于最大化期望对数似然，实际用有限数据的平均值近似。
- **训练似然高不保证泛化或样本质量好**：模型容量、数据量、优化和评估目标都影响结果。

## 1. 用 KL 散度比较两个分布

离散分布的 KL 散度为：

$$
D_{\mathrm{KL}}(p\|q)
=\sum_x p(x)\log\frac{p(x)}{q(x)}
=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right].
$$

连续密度将求和替换为积分。若 $p$ 在某些区域有正概率而 $q$ 在该区域为零，KL 为 $+\infty$；有限 KL 要求相应的支持关系与可积性。约定 $0\log(0/q)=0$。

KL 非负，等于零当且仅当两分布相同（连续情形为几乎处处相同）。它通常不对称，也不满足距离的全部公理。

<details>
<summary>为什么 KL 非负？</summary>

当 $p$ 关于 $q$ 绝对连续且表达式有定义时，利用 $-\log$ 的凸性：

$$
D_{\mathrm{KL}}(p\|q)
=\mathbb{E}_p\left[-\log\frac{q(x)}{p(x)}\right]
\geq-\log\mathbb{E}_p\left[\frac{q(x)}{p(x)}\right]
\geq0.
$$

最后一步利用 $q$ 在 $p$ 的支持集上的概率总量至多为 $1$。若两者具有相同支持，该期望等于 $1$；不满足必要支持关系时，KL 可以直接为无穷。

</details>

对离散分布，熵与交叉熵分别为：

$$
H(p)=-\mathbb{E}_p[\log p(x)],\qquad
H(p,q)=-\mathbb{E}_p[\log q(x)].
$$

因此，正确关系是 $D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)$。连续情形有相应的微分熵关系，但需要各项有定义。

## 2. 从分布拟合得到最大似然

取真实数据分布 $p_{\mathrm{data}}$ 与模型 $p_{\theta}$：

$$
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\theta})
=\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(x)]
-\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\theta}(x)].
$$

第一项与 $\theta$ 无关，所以在这些量有定义时：

$$
\arg\min_{\theta}D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\theta})
=\arg\max_{\theta}\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log p_{\theta}(x)].
$$

这个方向称为**正向 KL**。如果模型遗漏某个真实数据模式，使该模式上的模型概率趋近于零，就会受到很大惩罚，因此常呈现“覆盖模式”的倾向。反向 KL 的期望在模型分布下计算，受限模型拟合多峰分布时可能集中于部分模式；这些是典型行为，并非对任意模型族的绝对保证。

真实分布未知，但我们有独立同分布样本 $\mathcal{D}=\{x^{(1)},\ldots,x^{(N)}\}$。用蒙特卡洛平均近似期望，得到最大似然估计（MLE）：

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_{\theta}\ell_N(\theta),\qquad
\ell_N(\theta)=\sum_{k=1}^{N}\log p_{\theta}(x^{(k)}).
$$

由于样本独立，数据似然为 $L_N(\theta)=\prod_{k=1}^{N}p_{\theta}(x^{(k)})$。对数单调递增，因此最大化似然与最大化对数似然等价。实践中通常最小化**平均负对数似然（NLL）**：

$$
\widehat{\mathcal{L}}_{\mathrm{NLL}}(\theta)
=-\frac{1}{N}\ell_N(\theta).
$$

似然是“固定观测、把参数看作变量”的函数，不是参数的概率分布。对连续数据，这里乘的是概率密度，而非观测精确落在某个点的概率。

## 3. 两个可手算的例子

### 硬币：频率就是伯努利参数的 MLE

若 $x^{(k)}\in\{0,1\}$，$p_{\theta}(X=1)=\theta$，观察到 $N_1$ 次正面、$N_0$ 次反面，则：

$$
\ell_N(\theta)=N_1\log\theta+N_0\log(1-\theta),\qquad
\hat\theta=\frac{N_1}{N_1+N_0}.
$$

两类都出现时，由求导及凹性得到唯一内部最大值；全为正面或全为反面时，最大值在边界。课件示例 $H,H,T,H,T$ 的估计为 $3/5$。

### 高斯均值：已知方差下得到样本平均

设 $x^{(k)}\sim\mathcal{N}(\mu,\sigma^2)$，方差 $\sigma^2>0$ 已知，则：

$$
\ell_N(\mu)=C-\frac{1}{2\sigma^2}\sum_{k=1}^{N}(x^{(k)}-\mu)^2,\qquad
\hat\mu=\frac{1}{N}\sum_{k=1}^{N}x^{(k)}.
$$

因此，在固定方差高斯观测模型下，最大似然等价于最小化平方误差。损失函数的形式对应着具体的概率假设。

## 4. 复杂模型：分解目标，再用梯度优化

贝叶斯网络或自回归模型都可以将联合对数似然写为局部项的和。例如：

$$
\ell_N(\theta)=\sum_{k=1}^{N}\sum_{i=1}^{n}
\log p_{\theta}(x_i^{(k)}\mid x_{<i}^{(k)}).
$$

对不共享参数的离散 CPT，局部 MLE 可由条件频率得到；若某个父节点配置从未出现，对应条件概率不能仅靠这些数据确定。神经网络参数通常没有闭式解，需反向传播与数值优化。

| 采用的目标 | 一步更新 |
| --- | --- |
| 最大化 $\ell_N$ | $\theta\leftarrow\theta+\eta\nabla_{\theta}\ell_N$ |
| 最小化平均 NLL | $\theta\leftarrow\theta-\eta\nabla_{\theta}\widehat{\mathcal{L}}_{\mathrm{NLL}}$ |

两者分别是梯度上升与梯度下降，注意符号和目标的缩放。仅令梯度等于零并不能保证找到全局最优；非凸网络通常只能得到某个优化结果。

数据较多时，均匀抽取大小为 $B$ 的小批量 $\mathcal{B}$，使用：

$$
\widehat g_{\mathcal B}(\theta)
=-\frac{1}{B}\sum_{k\in\mathcal B}\nabla_{\theta}\log p_{\theta}(x^{(k)}).
$$

在固定参数、均匀抽样及可交换微分与期望等通常条件下，它是全数据平均 NLL 梯度的无偏估计。增大小批量一般可以降低估计方差，但不保证每一步都降低损失。

## 5. 样本平均为何有用，又为何会过拟合？

对固定 $\theta$，令 $f_{\theta}(x)=-\log p_{\theta}(x)$。若样本独立同分布且相关矩存在，则蒙特卡洛平均满足：

$$
\mathbb{E}\left[\frac{1}{N}\sum_{k=1}^{N}f_{\theta}(x^{(k)})\right]
=\mathbb{E}_{p_{\mathrm{data}}}[f_{\theta}(x)],\qquad
\operatorname{Var}\left[\frac{1}{N}\sum_{k=1}^{N}f_{\theta}(x^{(k)})\right]
=\frac{\operatorname{Var}(f_{\theta}(X))}{N}.
$$

大数定律进一步说明样本平均随数据量增加趋近总体期望。但训练出的 $\hat\theta$ **依赖于同一批数据**，因此不能把“固定参数时估计无偏”直接解释为“训练损失是泛化误差的无偏估计”。

总体风险与经验风险分别为：

$$
R(\theta)=\mathbb{E}_{p_{\mathrm{data}}}[\mathcal L(x;\theta)],\qquad
\widehat R_N(\theta)=\frac{1}{N}\sum_{k=1}^{N}\mathcal L(x^{(k)};\theta).
$$

| 现象 | 含义 | 常见应对 |
| --- | --- | --- |
| 欠拟合 / 高偏差 | 模型族过于受限，训练与验证表现都不足 | 改善表示、提高表达能力，并检查优化是否充分 |
| 过拟合 / 高方差 | 对当前训练样本过于敏感，训练好而验证差 | 更多数据、适当容量、正则化与验证集选择 |

“偏差—方差权衡”是理解统计估计的框架。经典平方误差的严格分解包含偏差平方、方差与不可约噪声；不能把同一个分解公式直接套到任意 NLL，也不能断定所有深度模型的误差曲线都呈简单 U 形。

课件的 VC 泛化界用于说明“容量与数据量影响泛化”。具体界需要指定损失、假设空间和概率条件；通常无界的 NLL（包括离散与连续模型）不能直接套用面向有界损失的简单形式。

实际阅读训练结果时，至少分开查看训练 NLL、独立验证 NLL 与生成样本。高似然、感知质量和有意义的表示相关，但不是同一个评价目标。

## 自检

1. **为什么 MLE 对应正向 KL？** 期望在真实数据分布下计算，去掉与参数无关的真实分布熵项即可。
2. **梯度上升和梯度下降是否矛盾？** 不矛盾；前者最大化对数似然，后者最小化其相反数。
3. **训练 NLL 下降就说明泛化更好吗？** 不能据此确定；需要独立验证数据，并检查实际关注的生成或表示指标。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文。

- [Lecture 4: Maximum Likelihood，45 页](../temp/Lecture4_Maximum_Likelihood.pdf#page=9)：第 9–20 页介绍模型族、KL 与总体目标，第 22–33 页介绍蒙特卡洛估计、MLE 示例和梯度优化，第 35–43 页介绍风险、过拟合与偏差—方差权衡。
- 课件补充：[Stanford CS 236 Lecture 4](https://www.youtube.com/watch?v=bt3dqcbMLa0&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8&index=4)、[Cornell CS 6785 Lecture 4](https://www.youtube.com/watch?v=-OjBWNmXYyg&list=PL2UML_KCiC0UPzjW9BjO-IW6dqliu9O4B&index=4)，以及 *Deep Generative Modeling* Appendix Section B。
- 衔接：[自回归模型](03-autoregressive-models.md)将 NLL 化为逐位置条件预测；潜变量模型则需要进一步处理边缘似然中的积分。

---

← [上一章：自回归模型](03-autoregressive-models.md) · [目录](../README.md) · [下一章：潜变量模型与 VAE](05-latent-variable-models-and-vae.md) →

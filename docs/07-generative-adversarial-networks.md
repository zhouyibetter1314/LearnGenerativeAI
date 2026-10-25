# 07 · 生成对抗网络

[返回首页](../README.md) · [符号约定](notation.md)

> 不直接计算生成样本的似然，而是让生成器与判别器对抗，用样本学习数据分布。

## 先抓住这几件事

- GAN 的生成器定义了一个可采样的隐式分布，通常不提供可直接计算的密度。
- 原始 minimax GAN 在理想最优判别器下对应 JS 散度；实际训练采用交替更新，常用非饱和生成器损失。
- WGAN 改变分布比较方式；条件 GAN、图像翻译与 StyleGAN 则展示了对抗损失和架构的扩展。

## 1. 为什么考虑不依赖似然的训练？

最大似然是有原则的分布学习方法，但**测试似然、视觉质量与样本多样性并非同一个评价维度**。课件给出如下混合分布：

$$
p_{\theta}(x)=0.01p_{\mathrm{data}}(x)+0.99p_{\mathrm{noise}}(x).
$$

其样本有 $99\%$ 来自噪声，但对任意满足 $p_{\mathrm{data}}(x)>0$ 的数据点：

$$
\log p_{\theta}(x)\geq\log p_{\mathrm{data}}(x)-\log100.
$$

这说明每个样本的期望对数似然差距可被一个常数控制；高维数据中，按维度平均的差距甚至可能很小。反过来，单纯复制训练样本可以看起来逼真，却没有证明模型能泛化。

一种替代思路是**双样本比较**：给出真实样本与模型样本，让分类器学习区分两者。有限样本下未能区分，只说明当前检验或分类器没有发现差异，不能证明两个分布完全相同。

## 2. 两个网络与原始目标

生成器 $G_{\theta}$ 将简单噪声映射为数据：

$$
z\sim p_Z,\qquad \tilde x=G_{\theta}(z).
$$

映射不要求可逆，潜变量也不必与数据同维。它诱导生成分布 $p_{\theta}$；即使该分布没有方便的密度表达式，也可以通过 $G_{\theta}$ 采样。

判别器 $D_{\phi}(x)$ 输出“样本来自真实数据”的概率。理论上允许 $D(x)\in[0,1]$；实际 sigmoid 网络的输出位于 $(0,1)$。取真实、生成两类等权，原始 minimax 目标为：

$$
\min_{\theta}\max_{\phi}V(\theta,\phi),\qquad
V=\mathbb E_{x\sim p_{\mathrm{data}}}[\log D_{\phi}(x)]
+\mathbb E_{z\sim p_Z}[\log(1-D_{\phi}(G_{\theta}(z)))].
$$

判别器最大化正确分类的对数概率；生成器试图使假样本更难被识别。**判别器目标的两项之间是加号**。

## 3. 最优判别器与 JS 散度

固定生成器，在判别器表达能力无限且优化充分的理想条件下：

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_{\theta}(x)}.
$$

该式在分母为正的区域成立，密度需相对于共同的支配测度理解；等权真实与生成样本是该形式的前提。允许边界取值时约定 $0\log0=0$；若限制判别器只能输出开区间值，理论最优值需理解为上确界及端点极限。

令 $m=(p_{\mathrm{data}}+p_{\theta})/2$，JS 散度定义为：

$$
\operatorname{JS}(p_{\mathrm{data}},p_{\theta})
=\frac12D_{\mathrm{KL}}(p_{\mathrm{data}}\|m)
+\frac12D_{\mathrm{KL}}(p_{\theta}\|m).
$$

代入最优判别器：

$$
\boxed{\max_D V(G,D)=-\log4+2\operatorname{JS}(p_{\mathrm{data}},p_{\theta}).}
$$

自然对数下 $0\leq\operatorname{JS}\leq\log2$；当且仅当分布相同时为零，届时 $D^*=1/2$。JS 是对称散度，满足三角不等式的是 $\sqrt{\operatorname{JS}}$，而非一般的 JS 本身。

<details>
<summary>展开：上述结论的关键推导</summary>

对每个 $x$，判别器最大化 $a\log d+b\log(1-d)$，其中 $a=p_{\mathrm{data}}(x)$、$b=p_{\theta}(x)$。当 $a,b>0$ 时，求导令零得到 $d=a/(a+b)$；仅一项为正时，最优值取相应端点。因此：

$$
V(G,D^*)=
\int p_{\mathrm{data}}\log\frac{p_{\mathrm{data}}}{2m}
+\int p_{\theta}\log\frac{p_{\theta}}{2m}
=-2\log2+2\operatorname{JS}(p_{\mathrm{data}},p_{\theta}).
$$

这是**原始 minimax 目标在最优判别器下**的结论，不能直接等同于有限网络、未优化充分或改用其他生成器损失时的训练过程。

</details>

## 4. 实际训练：交替更新与非饱和损失

每次迭代先采样真实数据和噪声，然后交替更新：

1. 固定 $G_{\theta}$，最小化判别器二元交叉熵 $L_D=-V$。
2. 固定 $D_{\phi}$，更新生成器；梯度仍需穿过判别器传回生成器。
3. 重复若干轮，监控样本质量、多样性及训练稳定性。

原始生成器损失为 $L_G^{\mathrm{MM}}=\mathbb E_z\log(1-D_{\phi}(G_{\theta}(z)))$。早期若判别器很容易识别假样本，$D(G(z))\approx0$，该目标可能提供较弱梯度。

常用的**非饱和损失（non-saturating loss）**为：

$$
L_G^{\mathrm{NS}}=-\mathbb E_{z\sim p_Z}\log D_{\phi}(G_{\theta}(z)).
$$

它直接鼓励假样本被判为真，改变梯度形状，且保持理想分布匹配的平衡点；它不再是上一节同一个 minimax 生成器目标。

常见困难包括梯度不稳定、双方损失振荡，以及**模式坍塌**：许多噪声输入只生成少数几类样本。逼真并不意味着覆盖了所有真实模式。JS 在某些受限模型族中可能偏向拟合部分模式，但模式坍塌还与网络、优化及训练动态有关，不能仅归因于一种散度。

## 5. WGAN：从分类差异到搬运距离

### 5.1 为什么 JS 可能缺乏距离信息？

若两个分布的支撑完全不相交，JS 为 $\log2$，无法反映两者在空间中究竟相距多远。例如：

$$
\operatorname{JS}(\delta_a,\delta_b)=\log2\quad(a\neq b),
\qquad W_1(\delta_a,\delta_b)=\|a-b\|_2.
$$

Wasserstein 距离将一个分布看作需要搬运到另一分布的“质量”，代价为“质量乘移动距离”。

### 5.2 最优传输与对偶形式

对具有有限一阶矩的分布：

$$
W_1(p,q)=\inf_{\gamma\in\Pi(p,q)}
\mathbb E_{(X,Y)\sim\gamma}\|X-Y\|_2.
$$

$\Pi(p,q)$ 是所有边缘分布分别为 $p,q$ 的联合分布，即运输方案的集合。离散情况下，运输矩阵满足 $\gamma_{ij}\geq0$、$\sum_j\gamma_{ij}=p_i$、$\sum_i\gamma_{ij}=q_j$，并最小化 $\sum_{ij}\gamma_{ij}d(i,j)$。

Kantorovich–Rubinstein 对偶将运输问题改写为：

$$
W_1(p,q)=\sup_{\|f\|_{\mathrm{Lip}}\leq1}
\left(\mathbb E_p f(X)-\mathbb E_q f(Y)\right).
$$

这里的 1-Lipschitz 约束与所选欧氏距离一致：$|f(x)-f(y)|\leq\|x-y\|_2$。

### 5.3 Critic 的目标与 Lipschitz 约束

WGAN 用实值函数 $f_{\phi}$ 作为 **critic**，无需 sigmoid，也不输出真假概率：

$$
\max_{\phi}\;\mathbb E_{x\sim p_{\mathrm{data}}}f_{\phi}(x)
-\mathbb E_z f_{\phi}(G_{\theta}(z)),\qquad
\min_{\theta}\;-\mathbb E_z f_{\phi}(G_{\theta}(z)).
$$

最大化要求 critic 受 1-Lipschitz 约束。课件介绍两种实现思路：

- **权重裁剪**：限制网络权重范围，操作简单，但会限制 critic 的表达能力；并不自动精确实现最优的 1-Lipschitz 函数族。
- **梯度惩罚**：在真实与生成样本之间采样插值点，正则化输入梯度。

以 WGAN-GP 的常用形式为例：

$$
\hat x=\epsilon x+(1-\epsilon)\tilde x,\quad\epsilon\sim\mathcal U(0,1),\qquad
L_{\mathrm{GP}}=\lambda\mathbb E_{\hat x}
\left(\|\nabla_{\hat x}f_{\phi}(\hat x)\|_2-1\right)^2.
$$

critic 最小化 $\mathbb E f(\tilde x)-\mathbb E f(x)+L_{\mathrm{GP}}$。这是对采样区域的软约束，并非全空间 Lipschitz 性的严格证明。WGAN 改善目标的几何性质，但仍需合适的架构和优化，不能保证消除模式坍塌。

## 6. 对抗损失可以与其他任务组合

对条件 $c$，条件生成器写为 $G_{\theta}(z,c)$，判别器可同时观察 $x,c$。常用组合为：

$$
L_G=L_{\mathrm{adv}}+\lambda_{\mathrm{rec}}L_{\mathrm{rec}}
+\lambda_{\mathrm{perc}}L_{\mathrm{perc}}.
$$

重建损失约束与目标的对应关系；感知或对抗损失强调特征与分布层面的相似。更强的视觉真实感不等于逐像素误差一定更小。

| 课件案例 | 主要思路 | 需要辨清的关系 |
| --- | --- | --- |
| SRGAN、图像修复 | 重建 / 内容约束加对抗损失 | 纹理逼真与 PSNR 可有取舍 |
| VQ-GAN | 给离散视觉自编码器加入感知与对抗训练 | 改善视觉 token 重建，后续仍可在 token 上训练自回归模型 |
| Pix2Pix | 用成对输入输出训练条件 GAN | 判别器观察输入与输出的组合 |
| CycleGAN | 两个方向的映射加循环一致性 | 不需要成对样本，但不能由此保证唯一、语义正确的对应 |
| StarGAN | 用一个条件模型处理多个域 | 域条件、域分类与重建约束共同作用 |
| ADDA | 对齐源域和目标域的特征分布 | 生成对象可为特征，而不一定是图像 |

CycleGAN 的循环一致性约束可写为：

$$
L_{\mathrm{cyc}}=
\mathbb E_{x\sim p_X}\|F(G(x))-x\|_1
+\mathbb E_{y\sim p_Y}\|G(F(y))-y\|_1.
$$

它与两个方向的对抗损失结合：对抗损失匹配目标域分布，循环约束减少任意映射的自由度。

## 7. 从高分辨率生成到可控生成

增加输出分辨率会同时加重优化、全局结构与局部纹理建模的困难。课件梳理了以下架构路线：

- **Progressive Growing GAN**：逐步增加生成器和判别器的分辨率，先学粗结构，再加入细节。
- **StyleGAN**：将潜变量映射到中间空间，并在不同层注入样式；额外噪声建模随机细节，使高层属性与局部变化更容易分开控制。
- **StyleGAN2**：系统修改结构与归一化相关设计，减少水滴状等伪影，改善图像质量。
- **后续阅读**：StyleGAN3 的抗混叠设计，以及文本条件、三维与大规模多样数据上的 GAN 扩展。

这些方法说明：损失函数、生成器架构与训练策略共同决定结果。不能仅凭“采用 WGAN”或“扩大网络”就推断训练稳定性与可控性。

## 自检

1. **判别器输出 $1/2$ 能否证明训练成功？** 不能；也可能是判别器能力不足或未训练好，理论结论还需要其最优性等条件。
2. **为什么 GAN 不要求 Flow 的可逆结构？** 其目标只依赖真实和生成样本，不需要通过变量变换计算密度。
3. **WGAN 的 critic 与普通判别器有什么区别？** 前者是受 Lipschitz 约束的实值评分函数，后者通常是用交叉熵训练的真假概率模型。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文。以下采用 PDF 文件页码；Lecture 10 第 22 页起部分幻灯片印刷页码与文件页码相差 1。

- [Lecture 9 · 似然与样本比较](../temp/Lecture9_Generative_Adversarial_Networks_I.pdf#page=8)，第 8–20 页；[GAN 定义、最优判别器与 JS](../temp/Lecture9_Generative_Adversarial_Networks_I.pdf#page=23)，第 23–35 页；[训练性质与模式坍塌](../temp/Lecture9_Generative_Adversarial_Networks_I.pdf#page=37)，第 37–48 页。
- [Lecture 10 · WGAN](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf#page=9)，第 9–30 页；[对抗损失的应用](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf#page=32)，第 32–54 页；[GAN 架构扩展](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf#page=56)，第 56–65 页。
- 公式核对：Lecture 9 第 4 页及 Lecture 10 第 3 页的 MLE 对应 KL 方向应为 $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\theta})$；Lecture 10 第 28 页原始 GAN 判别器目标的第二项应取加号。JS 不等于 $D_{\mathrm{KL}}(p\|q)+D_{\mathrm{KL}}(q\|p)$。

课件核心阅读包括 Goodfellow 等，*Generative Adversarial Nets*（2014）；Theis 等，*A Note on the Evaluation of Generative Models*（2016）；Arjovsky 等，*Wasserstein GAN*（2017）；Gulrajani 等，*Improved Training of Wasserstein GANs*（2017）；Blau 与 Michaeli，*The Perception-Distortion Tradeoff*（2018）。教材对应 *Deep Generative Modeling* 第 8 章。

应用可沿 *Pix2Pix*、*CycleGAN*、*StarGAN*、*ADDA*、*Taming Transformers for High-Resolution Image Synthesis* 阅读；架构路线可沿 *Progressive Growing of GANs*、*StyleGAN*、*StyleGAN2*、*StyleGAN3* 阅读。Lecture 10 第 65 页还列出 Projected GANs、StyleGAN-NADA、StyleGAN-XL、EG3D、GET3D、StyleGAN-T、GigaGAN、Diffusion-GAN 与现代 GAN 基线。

---

← [上一章：归一化流](06-normalizing-flows.md) · [目录](../README.md) · [下一章：扩散模型](08-diffusion-models.md) →

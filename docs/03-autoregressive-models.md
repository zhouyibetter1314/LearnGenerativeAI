# 03 · 自回归模型

[返回首页](../README.md) · [符号约定](notation.md)

> 将联合分布写成“给定已有内容，预测下一部分”的条件分布乘积。

## 先抓住这几件事

- **分解是精确的，模型未必精确**：链式法则不近似；有限规模网络及其结构会限制可表示的分布。
- **自回归不等于循环神经网络**：RNN、CNN、Transformer 都可以实现同一种概率分解。
- **训练与生成使用不同的前缀**：训练通常输入真实前缀，生成逐步输入模型已经采样的前缀。

## 1. 先确定顺序，再建模条件分布

对数据 $x=(x_1,\ldots,x_n)$，固定一种顺序，令 $x_{<i}=(x_1,\ldots,x_{i-1})$：

$$
p_{\theta}(x)=\prod_{i=1}^{n}p_{\theta}(x_i\mid x_{<i}),\qquad
\log p_{\theta}(x)=\sum_{i=1}^{n}\log p_{\theta}(x_i\mid x_{<i}).
$$

第一项约定为 $p_{\theta}(x_1)$。每个条件分布非负且归一化，就能得到归一化的联合分布。通过从最后一个变量起依次求和或积分，可以验证这一点。

**顺序不必来自自然时间**。文本可以从左到右排列；图像可以按行扫描、按空间块划分或按尺度排列；每个 $x_i$ 也可以是一组变量。不同顺序的链式分解都成立，但可能使条件分布的学习难度不同。

这体现了**归纳偏置（Inductive Bias）**：模型结构、变量顺序、局部感受野和参数共享方式会影响学习与泛化。这里的“自回归”指变量间的生成依赖，不限定某一种网络架构。

## 2. 网络输出的是分布参数

以二值图像为例，$x_i\in\{0,1\}$。令网络输出：

$$
\pi_i(x_{<i})=p_{\theta}(X_i=1\mid x_{<i}),\qquad
p_{\theta}(x_i\mid x_{<i})=\pi_i^{x_i}(1-\pi_i)^{1-x_i}.
$$

$\pi_i$ 是一个概率，生成时还需要从 $\operatorname{Ber}(\pi_i)$ 采样。将概率直接当作像素值，或每步总取最大概率类别，都不等于从该联合分布采样。

| 变量类型 | 一种常见输出方式 |
| --- | --- |
| 二元变量 | sigmoid 输出伯努利参数 |
| $K$ 类离散变量 | softmax 输出 $K$ 个类别概率 |
| 连续变量 | 输出高斯等条件密度的参数 |

例如，8 位灰度像素可建模为 256 类变量。RGB 图像还需明确像素顺序和通道顺序，确保条件输入符合约定的生成顺序。

## 3. 从简单模型到共享网络

### FVSBN：用逻辑回归表示条件概率

全可见 sigmoid 信念网络（FVSBN）使用：

$$
\pi_i(x_{<i})=\sigma\left(b_i+\sum_{j<i}w_{ij}x_j\right).
$$

共有 $n$ 个偏置和 $n(n-1)/2$ 个连接权重，参数量为 $n(n+1)/2$。相较完整条件概率表所需的 $2^n-1$ 个自由参数，它更紧凑，但只能表达条件对数几率对历史输入线性的分布。

### NADE：非线性条件分布与参数共享

神经自回归分布估计器（NADE）先计算隐藏表示，再输出每个条件分布：

$$
h_i=g\left(c+W_{:,<i}x_{<i}\right),\qquad
\pi_i=\sigma\left(b_i+V_{i,:}h_i\right).
$$

输入到隐藏层的权重 $W$ 与偏置 $c$ 在不同位置共享，输出权重可随位置改变。设隐藏维数为 $d_h$，参数量为 $O(nd_h)$；利用前缀累加更新隐藏层预激活，可避免重复计算整个前缀。

### MADE：在一个网络内施加掩码

掩码自编码分布估计器（MADE）限制网络中的连接，使第 $i$ 个概率输出只能依赖 $x_{<i}$，不能通过任何直接或间接路径看到 $x_i$ 或未来变量。

它的外形类似自编码器，但目标是预测**尚未看到的变量**；普通自编码器则根据输入重建输入。仅有重建函数并不会自动定义一个可采样、归一化的数据分布。

> 掩码是对计算依赖的严格约束。不能因为课件将其称为“imposed”，就把它理解成允许少量未来信息泄漏的软约束。

## 4. 架构不同，因果约束相同

这里的“因果”指计算图中不能读取未来位置，不自动表示现实世界的因果关系。

| 架构 | 如何使用历史 | 如何维持自回归约束 | 主要代价 |
| --- | --- | --- | --- |
| RNN / PixelRNN | 递归更新隐藏状态 | 只向后传递已有输入的信息 | 隐藏状态压缩历史；训练也有时间递归 |
| 因果 CNN / WaveNet | 卷积聚合历史邻域 | 因果卷积、输入移位；扩张卷积扩大感受野 | 有限深度下感受野有限 |
| PixelCNN | 二维卷积聚合已生成像素 | 空间及通道掩码 | 掩码设计可能造成上下文盲区 |
| Transformer / GPT / imageGPT | 注意力访问历史表示 | 输入移位与因果注意力掩码 | 标准全注意力的序列开销较大 |

同一个联合分布可以由一个共享网络同时输出所有位置的条件分布参数。关键不在“是否一次输入整条序列”，而在于**每个输出实际上能依赖哪些输入**。

## 5. 生成、似然计算与训练要分开看

### 生成：祖先采样

1. 从 $p_{\theta}(x_1)$ 采样 $x_1$。
2. 对 $i=2,\ldots,n$，输入已采样的 $x_{<i}$，从 $p_{\theta}(x_i\mid x_{<i})$ 采样 $x_i$。
3. 将各部分组合为完整样本。

在通常的单变量分解中，后一步需要前一步的采样结果，因此生成具有顺序依赖。参数共享和缓存能降低重复计算，却不会自动消除这种依赖。

### 似然计算：观测已经完整

给定完整样本时，所有真实前缀都已知。只要局部条件概率可计算，就能精确计算模型的联合对数似然，不必对缺失变量求和或积分。“精确”指相对于已定义的模型，**不表示模型等于真实分布**。

### 训练：Teacher Forcing

对训练数据 $\{x^{(k)}\}_{k=1}^{N}$，负对数似然为：

$$
\mathcal{L}_{\mathrm{AR}}(\theta)
=-\frac{1}{N}\sum_{k=1}^{N}\sum_{i=1}^{n}
\log p_{\theta}\left(x_i^{(k)}\mid x_{<i}^{(k)}\right).
$$

这里每一项的条件输入都来自**真实样本**，即 Teacher Forcing。二元输出对应二元交叉熵，类别输出对应类别交叉熵，详见[最大似然](04-maximum-likelihood.md)。

例如，训练“我／喜欢／数学”时，以“我”预测“喜欢”，以“我／喜欢”预测“数学”；生成时第二个词若采样为“学习”，第三步就会以“我／学习”为前缀。

| 场景 | 使用的前缀 | 位置间是否可以并行 |
| --- | --- | --- |
| 生成 | 模型已采样的结果 | 通常需要顺序进行 |
| 训练 / 已知样本的似然计算 | 真实观测值 | MADE、因果 CNN、Transformer 可并行计算多个位置；RNN 仍有递归依赖 |

Teacher Forcing 直接计算上述似然目标，**不需要对离散采样过程反向传播**。因此，不应把标准自回归训练理解成“先完整采样一遍，再穿过所有采样步骤求导”。

训练看到真实前缀，生成看到模型前缀；模型不完美时，两者可能出现分布偏移和误差积累，称为曝光偏差（Exposure Bias）。这不意味着 Teacher Forcing 对最大似然目标引入了额外的有偏近似。

## 6. 这类模型适合什么？

优势是联合概率结构清楚、可直接进行似然训练，且能自然表示可变长度或条件序列。主要限制是生成的顺序代价，以及变量顺序、局部感受野、表示粒度对建模效果的影响。

自回归也不只适用于离散 token：连续变量可以使用条件密度；图像可以先转为潜在 token 再生成。课件后续阅读中的 VQ、尺度顺序和自回归扩散，都是在修改表示、分组方式或局部条件分布。

## 自检

1. **链式分解本身是否损失信息？** 不损失；选择受限的条件分布族才可能引入表示误差。
2. **为何训练能并行而生成通常不能？** 训练中的真实前缀均已知，生成中的前缀需要逐步获得；具体并行能力还取决于架构。
3. **输出 $0.8$ 与生成二元值 $1$ 有何不同？** 前者是条件分布参数，后者是按该分布采样或按某种决策规则得到的取值。

## 来源与延伸阅读

原始课件保存在本地 `temp/`，在线阅读可直接使用本文。

- 主要依据：[更新版 Lecture 3，96 页](../temp/Lecture3_Autoregressive_Models_update260917.pdf#page=13)：第 13–28 页介绍分解与归纳偏置，第 30–51 页介绍 FVSBN、NADE、MADE，第 53–73 页介绍架构，第 75–93 页介绍生成与训练。
- 版本核对：[旧版 Lecture 3，88 页](../temp/Lecture3_Autoregressive_Models.pdf#page=13)。更新版增加了 RNN / PixelRNN、PixelCNN 结果及复杂度说明，并修正补充视频链接；旧版第 44 页关于高效计算的提示用于补充 NADE 的前缀复用说明。重复内容以更新版为主。
- 课件补充：[Stanford CS 236 Lecture 3](https://www.youtube.com/watch?v=tRArbBf-AbI&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8&index=3)、[Cornell CS 6785 Lecture 3](https://www.youtube.com/watch?v=Y3cJFaM8w2w&list=PL2UML_KCiC0UPzjW9BjO-IW6dqliu9O4B&index=3)，以及 *Deep Generative Modeling* Section 3。

<details>
<summary>课件列出的论文阅读方向</summary>

- 基础模型：Bengio 与 Bengio，*Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks*（1999）；Larochelle 与 Murray，*The Neural Autoregressive Distribution Estimator*（2011）；Germain 等，*MADE: Masked Autoencoder for Distribution Estimation*（2015）。
- 图像与音频架构：van den Oord 等，*Pixel Recurrent Neural Networks*（2016）、*WaveNet: A Generative Model for Raw Audio*（2016）；Chen 等，*Generative Pretraining from Pixels*（2020）。
- 自回归与流：Kingma 等，*Improved Variational Inference with Inverse Autoregressive Flow*（2016）。
- 表示与 tokenizer：*Language Model Beats Diffusion — Tokenizer is Key to Visual Generation*；*Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*；*An Image is Worth 32 Tokens for Reconstruction and Generation*；*Zero-Shot Text-to-Image Generation*；*Finite Scalar Quantization: VQ-VAE Made Simple*。
- 与扩散结合：*Autoregressive Image Generation without Vector Quantization*；*Transfusion*；*Show-o*；*Autoregressive Diffusion Models*；*Diffusion Forcing*。

</details>

---

← [上一章：概率建模](02-probabilistic-modeling.md) · [目录](../README.md) · [下一章：最大似然与泛化](04-maximum-likelihood.md) →

# 课件来源、覆盖范围与勘误

[返回首页](../README.md) · [符号约定](notation.md)

本笔记依据课程 *Generative Models & AI4S Applications*（刘云帆）的本地课件整理。`temp/` 中共有 **13 份 PDF，覆盖第 0–11 讲**；第 3 讲包含两个版本。下表页数与页码均按 PDF 文件页序计数，个别幻灯片印刷页码可能不同。

原始 PDF 保存在被 `.gitignore` 排除的本地 `temp/`，下列课件链接在本地有效；公开仓库中的笔记不依赖 PDF 才能阅读。这里只整理已有课件，不将路线图或阅读清单中的后续课程描述成已经讲授的正文。

## 1. 文件与知识章节的对应

| 本地课件 | 页数 | 整理位置与重点 |
| --- | --- | --- |
| [Lecture 0 · Overview](../temp/Lecture0_Overview.pdf) | 13 | [01 导论](01-introduction.md)：先修知识、课程范围、模型路线；教材见本页 |
| [Lecture 1 · Introduction](../temp/Lecture1_Introduction.pdf) | 41 | [01 导论](01-introduction.md)：建模目标、用途、挑战、历史与 AI4S 示例 |
| [Lecture 2 · Probabilistic Modeling](../temp/Lecture2_Probabilistic_Modeling.pdf) | 43 | [02 概率建模](02-probabilistic-modeling.md)：维度灾难、条件独立、贝叶斯网络、分类与参数化 |
| [Lecture 3 · Autoregressive Models，原版](../temp/Lecture3_Autoregressive_Models.pdf) | 88 | [03 自回归](03-autoregressive-models.md)：与更新版逐项核对，补充 NADE 前缀复用提示 |
| [Lecture 3 · Autoregressive Models，260917 更新版](../temp/Lecture3_Autoregressive_Models_update260917.pdf) | 96 | [03 自回归](03-autoregressive-models.md)：以此为主，含 FVSBN、NADE、MADE、RNN/CNN/Transformer 与训练生成 |
| [Lecture 4 · Maximum Likelihood](../temp/Lecture4_Maximum_Likelihood.pdf) | 45 | [04 最大似然](04-maximum-likelihood.md)：KL、MLE、蒙特卡洛、梯度优化、泛化与偏差方差 |
| [Lecture 5 · Latent Variable Models I，260924 更新版](../temp/Lecture5_Latent_Variable_Models_I_updata260924.pdf) | 63 | [05 潜变量与 VAE](05-latent-variable-models-and-vae.md)：GMM、边缘化、重要性采样、ELBO 与变分推断 |
| [Lecture 6 · VAE II，260929 更新版](../temp/Lecture6_Variational_Autoencoder_II_update260929.pdf) | 78 | [05 潜变量与 VAE](05-latent-variable-models-and-vae.md)：摊销推断、重参数化、β-VAE、坍塌、VQ-VAE |
| [Lecture 7 · Normalizing Flow I](../temp/Lecture7_Normalizing_Flow_I.pdf) | 39 | [06 Flow](06-normalizing-flows.md)：密度变换、组合、可逆性与 Jacobian |
| [Lecture 8 · Normalizing Flow II](../temp/Lecture8_Normalizing_Flow_II.pdf) | 56 | [06 Flow](06-normalizing-flows.md)：NICE、Real NVP、MAF/IAF 与概率密度蒸馏 |
| [Lecture 9 · GAN I](../temp/Lecture9_Generative_Adversarial_Networks_I.pdf) | 49 | [07 GAN](07-generative-adversarial-networks.md)：似然与质量、双样本比较、最优判别器、JS、非饱和与模式坍塌 |
| [Lecture 10 · GAN II](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf) | 67 | [07 GAN](07-generative-adversarial-networks.md)：WGAN、梯度惩罚、图像翻译、域适应与 StyleGAN 路线 |
| [Lecture 11 · Diffusion Models](../temp/Lecture11_Diffusion_Models.pdf) | 84 | [08 扩散](08-diffusion-models.md)：DDPM 正反过程、变分目标、噪声预测、训练采样与层次 VAE 联系 |

## 2. 如何取舍原始内容？

- **保留主线**：定义、建模假设、关键公式、训练与生成流程、方法关系与限制。
- **合并重复**：相邻讲次的回顾、同一推导的逐步动画页、重复图示，合为连续解释。
- **分层阅读**：较长推导放入折叠区；大量模型或论文名称保留为延伸方向。
- **不纳入知识正文**：联系方式、课程群、签到、作业截止时间与评分规则。这些属于课程管理信息，并非可复用的知识总结。
- **区分讲授与预告**：AI4S 应用在当前资料中主要是示例；Score/SDE、Flow Matching、引导与评估等主题未在本课件集独立展开。

原第 3 讲的 88 页版本与 96 页更新版讲授同一主线。更新版补充 RNN/PixelRNN 与 PixelCNN 结果、复杂度说明，并修正补充视频链接；正文以更新版为主，避免重复形成两章。

## 3. 公式勘误与严谨性说明

以下为核对原 PDF 后采用的修正或补充；笔记正文直接使用修正后的形式。

| 原位置 | 修正 / 需要补充的条件 |
| --- | --- |
| [Lecture 4，第 14 页](../temp/Lecture4_Maximum_Likelihood.pdf#page=14) | KL 与熵的正确关系为 $D_{\mathrm{KL}}(p\Vert q)=H(p,q)-H(p)$ |
| [Lecture 8，第 21 页](../temp/Lecture8_Normalizing_Flow_II.pdf#page=21) | Real NVP 逆映射应乘 $\exp(-a)$，对应正向缩放的倒数 |
| [Lecture 9，第 4 页](../temp/Lecture9_Generative_Adversarial_Networks_I.pdf#page=4)；[Lecture 10，第 3 页](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf#page=3) | MLE 对应 $D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta)$，不能交换 KL 方向 |
| [Lecture 10，第 28 页](../temp/Lecture10_Generative_Adversarial_Networks_II.pdf#page=28) | 原始 GAN 判别器最大化目标中，真实项与生成项之间应为加号 |
| Lecture 2 / 7 / 8 的高斯记号 | 统一第二参数为方差或协方差；Flow 的对数标准差与 VAE 的对数方差分别定义 |
| Lecture 2，第 32 页的线性边界示例 | 有线性边界不等于判别式建模；LDA 通常从类条件高斯与共享协方差出发 |
| Lecture 4 的泛化讨论 | 固定参数的蒙特卡洛无偏性不等于拟合后训练风险无偏；VC 界与偏差方差分解需要各自适用条件 |
| Lecture 11 的末端先验与噪声目标 | 有限日程的末端通常只是接近高斯；去掉时间权重后的噪声 MSE 不等于精确 NLL |

JS 的三角性质还需要保留平方根：成为距离的是 $\sqrt{\operatorname{JS}}$。一些公式的根号在纯文本提取时会丢失，因此换元、DDPM 与关键 KL 公式均回看了原页图像。

## 4. 教材与课程入口

以下入口来自第 0 讲及各讲末尾的推荐资料，作为继续阅读的线索；本仓库的正文依据本地课件整理。

| 资源 | 用途 |
| --- | --- |
| Jakub M. Tomczak，*Deep Generative Modeling*，第 2 版（2024）；[配套代码](https://github.com/jmtomczak/intro_dgm) | 主要模型族的系统教材与实现 |
| [The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/) | 概率基础、变分视角与扩散的后续阅读 |
| Goodfellow、Bengio、Courville，*Deep Learning* | 深度学习基础 |
| Bishop，*Pattern Recognition and Machine Learning*；Murphy，*Machine Learning: a Probabilistic Perspective* | 概率建模、统计学习与推断 |
| [Stanford CS 236](https://deepgenerativemodels.github.io/) | 生成模型课程与材料 |
| [Cornell CS 6785](https://kuleshov-group.github.io/dgm-website/) | 生成模型课程与材料 |
| [MIT 6.S978](https://mit-6s978.github.io/) | 生成模型结构与训练视角 |
| [MIT 6.S184](https://diffusion.csail.mit.edu/2026/)；[CMU 10-799](https://kellyyutonghe.github.io/10799S26/) | 后续 SDE/ODE、扩散与 Flow 主题 |

## 5. 按主题继续读哪些论文？

各章保留更具体的阅读方向，这里只给出起点。

| 主题 | 课件中的代表阅读 |
| --- | --- |
| 图像统计与先验 | Zoran 与 Weiss 的自然图像 patch / GMM 工作；*Deep Image Prior*；纹理合成与 PatchMatch |
| 自回归 | NADE；MADE；PixelRNN；WaveNet；imageGPT |
| 潜变量与 VAE | *Auto-Encoding Variational Bayes*；*Stochastic Backpropagation*；IWAE；β-VAE；VQ-VAE |
| Flow | NICE；Real NVP；MAF；IAF；Glow；Parallel WaveNet；FFJORD（延伸） |
| GAN | *Generative Adversarial Nets*；*A Note on the Evaluation of Generative Models*；WGAN；WGAN-GP；Pix2Pix；CycleGAN；StyleGAN 系列 |
| 扩散 | Sohl-Dickstein 等（2015）；DDPM（2020）；DDIM、Score SDE、Improved DDPM、Guidance、Latent Diffusion、EDM（延伸） |

---
title: Popularity prediction for marketer-generated content：A text-guided attention neural network for multi-modal feature fusion
date: 2025-03-22 14:44:00 +0800 #时间， 最后为时区北京 +0800
categories: [论文] #上级文档，下级文档
tags: [流行度预测, 多模态, text-guided attention]     # TAG
---

---

## 论文概况

研究问题：预测营销商生成内容（MGC）的受欢迎程度

提出方法：一种融合多模态特征的文本引导的注意力神经网络模型（TGANN）

模型特点：
1. 设计了一个基于过滤器的主题模型，以滤除噪声单词并从文本描述中提取主题功能。2. TGANN 集成主题模型学到的文本特征，以文本指导获取图像特征以及其他辅助特征（标签、标题）
2. 解决包含异质、多模态数据、文本和图像的噪声问题、多模态特征的融合机制的 MGC 的流行度预测问题
3. 为了减少图像中无关信息的影响，我们然后提出了一种文本引导的注意机制，以使用文本的主题功能来指导图像区域表示。
4. 为了确定每个主题的贡献和每个图像的贡献，Tgann 模型介绍了每种视觉方式和文本方式的注意力计算。

数据：两个现实世界数据集上进行的。分别从 taobao、autohome 中收集

实验结果：所提出的模型优于几种最新方法。

结论：我们的模型可以准确捕捉主题的注意力，图像关注和图像区域的关注。

研究展望：
1. MGC 受欢迎程度视为回归任务
2. 外部的一些信息（名人、其他社交媒体）可能影响流行度
3. 开展关于 MGC 中不同特征（如图像质量，文本内容中的主题）对用户参与度（点击数）的实证研究

意义：为营销人员和在线平台提供了重要的实践意义，例如估计在线广告活动的成功，创建更具吸引力的营销内容以及改进推荐系统。

---

## 引言



## 相关工作



## 模型

TGANN 整合了由主题模型、以文本特征为指导的图像特征、其他辅助特征（例如标签和标题），以增强预测性能。

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221046174.png)

### 问题定义

定义若干符号表示，描述问题为给定一组 MGC，提取特征后，学习一个函数去预测流行度得分。

假设存在一个包含 $D$ 个营销者生成内容（MGC）的集合。对于索引为 $i$ 的 MGC，其由文本描述 $d_i=\left\{w_{i, 1}, w_{i, 2}, \cdots, w_{i, n}, \cdots, w_{i, N_i}\right\}$ 表示，对应图像集 $I_i=$ $\left\{p_{i, 1}, p_{i, 2}, \cdots, p_{i, m}, \cdots, p_{i, M_i}\right\}$ ，其标签集为 $l_i$ ，标题信息为 $t_i$ ，作者信息为 $a_i$ ，时间信息为 $T_i$ 。令 $w_{i, n}$ 表示 $d_i$ 中的第 $n$ 个词，$p_{i, m}$ 表示 $I_i$ 中的第 $m$ 个图像。令 $N_i$ 为 $d_i$ 中的词数，$M_i$ 为 $I_i$ 中的图像数。

通过特征提取方法，我们获得 MGC $i$ 的不同模态的特征表示，记为 $F_i=$ $\left\{f_i^{\text {text }}, f_i^{\text {image }}, f_i^{\text {label }}, f_i^{\text {title }}, f_i^{\text {author }}, f_i^{\text {time }}\right\}$ 。按照先前的研究（Liao 等，2019；Xiong 等，2021），我们将流行度预测任务视为**分类任务**。我们将**总浏览数**离散化，并定义 $y_i \in\{0,1\}$ 来表示 MGC $i$ 的流行度分数。

基于上述符号，我们定义问题如下：给定 MGC $i$ ，我们的任务是学习一个函数 function ： $\left\{f_i^{\text {text }}, f_i^{\text {image }}, f_i^{\text {label }}, f_i^{\text {title }}, f_i^{\text {author }}, f_i^{\text {time }}\right\} \rightarrow y_i$ ，以预测其流行度分数。

> 点击次数/总浏览数是以某个值为界限，如果超过，那么就为 1，表示受欢迎，没超过就视为 0，表示不受欢迎。
> 
> 没有像其他论文那样，使用对数化。

### 特征提取

#### 文本特征

滤波主题模型 FBT，用于提取文本特征
![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221049181.png)
![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221424134.png)


#### 图像特征

使用预训练的 VGGNet16 模型提取每个图像的特征表示

1. 所有图像被调整为224×224像素
2. 对图像做区域划分，N=7*7 份
3. 获取每个图像区域的特征
4. **单张图像的特征就是每个块的图像特征的拼接**？  `7*7*512`
5. 通过全连接层，保持图像特征的维度和文本特征的维度相同


![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202503212150460.png)


#### 其他特征

- **标签和标题**：使用长短期记忆网络（LSTM）提取标签和标题的特征 $f_i^{title}$
- **作者信息**：使用作者的粉丝数、关注数和 MGC 数量来表示作者特征 $f_i^{author}$
- **时间信息**：将发布时间转换为小时、周、日、月和年五个子特征 $f_i^{time}$

### 文本引导的注意机制

文本引导的注意机制（text-guided attention）的核心思想是利用文本信息来指导模型在视觉数据中关注哪些区域。具体来说，模型会根据文本描述生成一个注意力图（attention map），这个注意力图会告诉模型在图像中哪些部分是与文本描述相关的，从而使得模型能够更加集中地处理这些关键区域。

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221049512.png)

### 用于特征融合的多模态紧凑双线性

使用多模态紧凑双线性（multimodal compact bilinear, MCB）池化方法融合视觉和文本特征。MCB 池化通过外积生成高维表示，然后使用线性模型 $L$ 进行学习。通过计数草图投影函数 $\Psi(\cdot)$ 将外积投影到低维空间，减少模型参数。最终融合表示为 
$$
C_i=L[\tilde{u}_{text}^i⊗\tilde{v}_{image}^i]
$$
上式中，$C_i$ ：最终融合表示，用于表示第 $i$ 个样本的多模态特征；$L$ ：线性模型，用于学习外积生成的高维表示；$\Psi(\cdot)$ ：计数草图投影函数（count sketch），用于将外积投影到低维空间；$\tilde{u}_{\text {text }}^i$ ：第 $i$ 个样本的文本特征向量；$\tilde{v}_{\mathrm{image}}^i$ ：第 $i$ 个样本的图像特征向量；$\otimes$ ：外积操作，用于生成高维表示。

### 预测

将融合文本和图像后的特征 $C_i$ 与其他辅助特征 $[f_i^{title}, f_i^{label}, f_i^{author}, f_i^{time}]$ 连接，得到全局特征 $f_{global}$

全局特征输入 sigmoid 函数预测流行度分数 $\hat{y}$：
$$
\hat{y} = sigmoid(W_g ​f_{global}​+b_g​)
$$

### 训练

使用交叉摘损失函数 $J$ 来衡量模型预测值 $\hat{y}_i$ 与真实值 $y_i$ 之间的差异：

$$
J=-\sum_{i=1}^S\left(y_i \log \hat{y}_i+\left(1-y_i\right) \log \left(1-\hat{y}_i\right)\right)
$$
上式中，$S$ ：训练集中的样本总数；$y_i$ ：第 $i$ 个样本的真实标签，取值为 0 或 1；- $\hat{y}_i$ ：第 $i$ 个样本的预测标签，取值范围为 $(0,1)$ 。

## 实验

首先，介绍实验细节，包括数据集，基准和评估指标。
然后，我们报告结果，即总体绩效和消融研究。
接下来，我们通过可视化进行定性分析。
最后，我们提出了几种重要的实践意义。

### 数据集

从 taobao.com 和 autohome.com 爬取数据。计算 MGC 在 autohome.com 和 taobao.com 中的点击次数后发现，MGC 点击次数分布遵守了高度偏斜的幂律分布（the power-law distributions, which are highly skewed）。
但是，TAOBAO.com 中 MGC 的点击远低于 Autohome.com 中的单击。主要原因是 Taobao.com 上的用户主要使用其搜索功能，而对 MGC 的关注较少。

因此，在淘宝的 MGC，我们将总体受欢迎程度分为“热”和“冷”，将截止点设置为 300 次点击。对于 autohome.com 中的 MGC，我们将整体流行度分为“热”和“冷”，将截止点设置为 10,000 次点击。

此外，我们发现“冷” MGC 的数量比“热” MGC 的数量大得多，从而导致效果不平衡。为了避免对“冷”类别的预测偏见，我们对所选样本数据使用 an undersampling technique 构建平衡数据集（Chawla 等，2004）。

我们为建议的模型进行预处理并标准化我们的原始数据集。具体来说，我们将 Jieba 工具应用于将文本描述和标题分割为有意义的中文单词。然后，对于文本描述和标题，我们删除了所有停用词。我们还删除了低频单词，其发生频率在文本描述中不超过 2％，从而确保主题结果不会受到异常单词的影响。此外，我们选择还包含图像和标签的 MGC。

表 1 总结了我们最终数据集的详细统计信息。为了训练我们提出的模型，我们将数据集的 80％随机分为训练集，10％ 作为验证集，10％ 作为测试集。

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221058580.png)

### baseline

选择用于流行度预测的传统的基于特征的方法、基于注意力机制的深度学习模型作为基准模型。

logistic 回归（LR）：LR 是最简单的分类模型，可提供基准性能。为了训练该模型，我们首先使用预训练的 VGGNET，并根据 VGGNET 中完整连接层的输出获得每个图像功能。然后，我们将每个 MGC 的所有图像特征的平均值作为 LR 模型的输入。此外，我们直接应用 FBT 模型学到的文档主题分布来表示每个 MGC 的文本描述。

支持向量机（SVM）：SVM（Chang＆Lin，2011 年）是一个广泛用于分类的模型。为了训练模型，我们考虑了类似于 LR 中包含的功能的多模式特征。此外，我们使用网格搜索方法来优化 SVM 的参数。 

LightGBM：LightGBM（Ke 等，2017）是合奏学习中有效的增强模型，它可以在保持学习决策树的准确性和减少数据实例的数量之间建立良好的平衡。在先前的工作之后（He 等，2019），我们将多个功能馈送到 LightGBM 中，以预测 MGC 的普及得分。

共同注意力网络（Co-attention Network, CAN）：最后一个强大的基线方法是从多模式学习的共同注意力网络中汲取灵感的。罐子是视觉问题回答和建议的最新模型（Lu 等，2016； Ma 等，2019）。但是，该模型可以简单地通过共同发入网络结合文本和视觉信息。为了适应我们的数据集，我们通过引入其他信息（例如标签和标题）来扩展模型。

### 评估指标

将受欢迎程度预测任务视为分类，因此我们使用三个标准指标，包括精度，召回和 F1 来评估模型性能。

### 性能比较

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221050311.png)


![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221050682.png)



![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221050925.png)



![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221053982.png)



![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221053165.png)


### 消融实验

进行消融实验以测试每个特征或模块的贡献

| 模型         | 解释                      |
| ---------- | ----------------------- |
| TGANN_nAF  | 无辅助特征（标签，标题，作者和时间信息）    |
| TGANN_nI   | 无图像特征，并删除 MCB 池化层       |
| TGANN_nTD  | 无文本特征，并删除 MCB 池化层       |
| TGANN_nTGA | 无 text-guided attention |


![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221051881.png)


![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221051255.png)


### 定性分析

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221057820.png)



![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221057795.png)



![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502221057677.png)


### 管理启示


## 结论和展望

结论：我们的模型可以准确捕捉主题的注意力，图像关注和图像区域的关注。

研究展望：
1. MGC 受欢迎程度视为回归任务
2. 外部的一些信息（名人、其他社交媒体）可能影响流行度
3. 开展关于 MGC 中不同特征（如图像质量，文本内容中的主题）对用户参与度（点击数）的实证研究
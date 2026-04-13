---
title: Transformer
date: 2025-02-21 20:08:00 +0800 
categories: [自然语言处理] 
tags: [transformer,attention]
math: true
---

**transformer 是一种基于 Encoder-Decoder 架构的模型，利用 Self-attn 机制来解决 seq2seq 任务。**

本文内容基于 Attention is all you need 论文，对于论文中的难点参考其他资料。

![](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502172056321.png)

## 工作流

本节对 Transformer 的训练推理工作流、Encoder 和 Decoder 工作流做出解释。

从整个过程来看，encoder 接收输入，串联执行，由最后一层 encoder 生成一个固定长度的上下文向量，传给每个 decoder 作为K、V。每个 decoder 之间是串联的，接收链子 encoder 的上下文向量和输入，由最后一个 decoder 生成 next token，并循环执行 decoder 模块获取后续的token。

```
X ─→ [Enc1]─→[Enc2]─→...─→[EncN]
                               │ Memory
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
           [Dec1]  ─→       [Dec2]  ─→  ... [DecM] ─→ Y
         (cross-attn)     (cross-attn)     (cross-attn)
```

![编码器-解码器架构](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502121047415.png)



### 训练推理工作流

区分 N 个 Encoder 模块和多头注意力机制，头数 h 和 Encoder 数不是一个东西，Encoder 模块包括多头注意力机制。

训练阶段，输入序列经过 N 个 Encoder 模块输出矩阵 $K$、$V$，这些矩阵包含了输入序列的全局信息，不会再重复生成。

第一个 Decoder 模块接收起始提示符经过基于掩码的多头注意力块，生成矩阵 $Q$，和来自 Encoder 模块的矩阵 $K$、$V$ 一起输入到多头注意力块，后面经过前馈全连接网络等模块，并将结果输出给第二个 Decoder 模块，重复过程，直到第 N 个模块输出。

训练过程可以并行，但是推理过程依赖于 Decoder 模块的上一步的输出。

## 模块详解

本节对 Transformer 中每个最小单位的模块做出解释。

### 输入及编码

流程：tokenizer -> embedding ->

以获得"你好世界！"这句话的嵌入为例，来理解 tokenizer 和 embedding

1. tokenizer 过程
   1. 切分：它会把句子切分成它认识的最小单位，即： ["你好", "世界", "！"]
   2. 查找、转换：查找这几个单位的对应数字序列，如：[1527, 2054, 102]。
2. embedding 过程
   1. 将数字序列中每个数字，进一步用一串数字（向量/高维空间中的某个点）来描述

tokenizer像是字典的索引页，索引页中记录了字典中所有的字、词的页码，embedding就是通过页码找到字词的具体含义

为什么不能将以上两步合为一步：如果要合为一步，就要求字典中记录每一个可能出现的字或词及其对应的向量，对字典覆盖范围要求太高。

如果拆分成两步，即便有索引页中没有的字或词，可以尝试进一步拆分，比如：找不到"transformer"，可以拆分为"trans"、"form"、"er"来分别寻找特征表示，然后合并以得到transformer的表征

#### 位置编码（Postional Encoding）

##### 理解

>位置编码： https://ai.feishu.cn/wiki/GmjKw7fkJiggxJkcR9zc0RGEnbh

自注意力机制具有「置换等变」性，表现为**无法捕捉顺序对语义的影响**，比如将"我喜欢你"和"你喜欢我"视为语义相同。

为了补充位置信息，位置编码的实现方式为：给输入 token 的向量拼接一个位置信息向量 $E$。

数学推导：令 $P$ 为**置换矩阵**（仅做交换两行初等变换的初等矩阵），$PX$ 表示对输入序列 $X$ 按行调整顺序，$X$ 为每个输入字符的行向量拼接得到，令 $A()$ 表示缩放点积注意力算子。

$$
\text{A}(PX)=
\operatorname{softmax}\left[\frac{\left(PX W_Q\right)\left(PX W_k\right)^{\top}}{\sqrt{d_k}}\right] (PV)=\text { softmax }\left(\frac{PQ K^{\top} P^{\top}}{\sqrt{d_k}}\right) PV=P\text { softmax }\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V=P\text{A}(X)
$$

根据上式的计算结果可知，调整输入序列中的顺序会导致输出序列中的顺序发生响应调整，不会影响每个输出元素的内部表征。

$$
\text{A}(P(X+E)) \neq P\text{A}(X+E)
$$

而当为每个输入 token 增加位置编码 $E$ 后，如上式所示，输出元素的表征发生变化，也就是不同位置的相同 token 携带的信息不同。

##### 位置编码的特点

* **区分性**：不同位置的编码要有差异
* **表达能力强**：捕捉位置的丰富信息，如：第几个token，与其他 token 的距离
* **可学习性**：对于变长输入或推理时超出训练长度，编码要有外推能力
* **高效性**：不能导致计算资源占用太多，比如在原词嵌入上拼接位置编码，会使得后续权重矩阵维度增加，增加计算资源。
* **平滑性**：位置编码与原词嵌入相加不会使结果偏离过远而破坏原有单词的语义信息，如正余弦位置编码数值范围为 $[-1,1]$。
* **兼容性**：能够与 token 的 embedding 拼接或相加，维度保持一致。

##### 常见的位置编码

为了让 transformer 捕捉到输入序列的位置信息，主要有两类方法：1、将位置信息融入词嵌入中（绝对位置编码）；2、微调自注意力模块，使有能力分辨不同位置的 token（相对位置编码）。

> https://kexue.fm/archives/8130

###### 固定正弦-余弦位置编码（绝对位置编码）

利用不同频率的正弦和余弦函数为每个位置生成一个独特的向量，然后将位置编码**加**到语义向量中。

计算第 $pos$ 个 token 中第 $2i$ 和第 $2i+1$ 维度的位置编码公式，如下所示：

$$
\begin{array} \\
PE(pos,2i)​=sin(\frac{pos​}{10000^{2i/d}}) \\ 
PE(pos,2i+1)=cos⁡(\frac{pos}{10000^{2i/d}})
\end{array}
$$

上式中，$pos$ 表示待计算位置编码的输入 token 在输入序列中的位置索引（从 0 开始）；$i$ 表示该 token 的位置编码维度索引（$0 \leq i \leq d_{model}/2-1$）；$10000$ 是一个经验超参数，用于确保编码唯一性；$d$ 表示语义 embedding 的总长度，一般为 512；$PE(pos,j)$ 表示第 $pos$ 个 token 的第 $j$ 维度的位置编码。

计算过程：假如计算**序列中第 $pos$ 个位置的 token** 的**第 $2i$ 个位置编码**，那么使用 sin 计算；如果计算第 $2i+1$ 个位置编码，那么使用 $cos$ 计算。最终得到每个位置编码的 512 维的位置编码。

**第 $pos+k$ 个位置编码是第 $pos$ 个位置编码的线性组合，这意味着位置编码中蕴含着输入 token 之间的距离信息。**

下面证明这一结论。假设第 $pos+k$ 个位置编码表示为：$PE(pos+k)$，暂时不考虑位置编码的维度索引。根据三角函数和差化积公式推导 $PE(pos+k)$ 和 $PE(pos)$ 的关系

$$
\begin{array} \\
\sin(pos+k)=\sin(pos) \; \cos(k) + \cos(pos) \; \sin(k) \\
\cos(pos+k)=\cos(pos) \; \cos(k) - \sin(pos) \; \sin(k) 
\end{array}
$$

整理为矩阵形式：

$$
\left[\begin{array}{c}
\sin (pos+k) \\
\cos (pos+k)
\end{array}\right]=\left[\begin{array}{cc}
\cos k & \sin k \\
-\sin k & \cos k
\end{array}\right]\left[\begin{array}{l}
\sin (pos) \\
\cos (pos)
\end{array}\right]
$$

进一步可以推导出所有位置编码之间的关系，等式右边即为块对角矩阵，以四个元素为一个块。


```python
import torch
import math
class PositionalEncoding(torch.nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		# 创建位置编码矩阵，大小为 [max_len, d_model]
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0))/ d_model)
		# 分奇偶计算
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		# x: tensor, shape [batch_size, seq_len, d_model]
		seq_len = x.size(1)
		x = x + self.pe[:, :seq_len]
		return x
```

###### 旋转位置编码（RoPE）

相对位置编码：对 $A$ 矩阵处理

RoPE 是目前应用较为广泛的一种相对位置编码。

三角函数能捕捉相对位置，但对远距离关系很弱。RoPE 通过构造 pair 显示建模相对位置，更加适合长序列。


### Encoder 模块

输入一系列向量，输出一系列向量，因为核心模块还是自注意力机制，所以输入和输出向量个数相等。

整个 Encoder 模块由 **6 个相同**的子 Encoder 模块顺序连接构成，第一个 Encoder 接收输入，经过多头注意力、前馈全连接网络、残差连接和正则化后，输出给下一个 Encoder 模块，最后一个 Encoder 模块输出 K、V 矩阵给 Decoder 模块。

多个 Encoder 层之间是按顺序串联的，第一层 Encoder 的输出会作为第二层的输入。

> **Encoder 中的模块和多头注意力机制有什么关系？**
> 每一层 Encoder 模块都包含：多头自注意力（Multi-Head Self-Attention）；前馈网络（Feed-Forward Network）

![[Pasted image 20251207090025.png]]

#### 自注意力及多头注意力机制

参考：[[2025-11-22-几种常见注意力机制]]

#### 前馈全连接网络

前馈全链接网络（Position-wise Feed-Forward Networks）用于变换数据维度，包括：两个全连接层和一个 RELU 激活层，数学上表示为：

$$
FFN(x)=max(0, W_1 x + b_1)W_2 + b_2
$$
上式中，$W_1 x + b_1$ 为第一层全连接层，$max(0, W_1 x + b_1)$ 为 RELU 激活函数，$max()W_2 + b_2$ 为第二层全连接层。$W_1$ 维度为 $d_{model} \times d_{ff}$，$W_2$ 维度为 $d_{ff} \times d_{model}$。

#### Add & Norm

这部分由残差连接（Residual Connection）和层归一化（Layer Norm）构成。

**残差连接**是一种常见的深度学习技巧，它将输出和其输入相连来实现，数学上表示为：

$$
Residual = x + Residual(x)
$$

上式中，$x$ 为残差连接层的输入，$Residual(x)$ 为残差连接层的输出。

作用：缓解深层网络中的梯度消失问题。

**层归一化**可以理解为对同一特征、同一个样本中不同维度计算均值和标准差，并归一化。

>为什么 transformer 使用层归一化：Transformer 处理的是文本序列，不同样本的序列长度不同。BatchNorm 需要在 batch 维度上对同一位置的样本求统计量，但序列长度不一致时，位置对齐本身就是问题，统计量的计算会非常不稳定甚至无意义。而 LayerNorm 只在单个样本内部计算，完全不依赖其他样本，天然适合变长序列。

>BatchNorm 是对不同样本不同特征的同一维度计算均值和标准差，并归一化。关于在 transformer 中选择 LayerNorm 的原因以及其他更优的归一化模块，可见论文：PowerNorm: Rethinking Batch Normalization in Transformers
>transformer 架构中的 LayerNorm 归一化模块存在优化的地方，可见论文：On Layer Normalization in theTransformer Architecture

作用：稳定训练过程、提高模型稳定性、减少梯度消失和爆炸

### Decoder 模块 / 自回归

问题：如果是从矩阵角度来思考，transformer 架构如何实现 mask 处理，得到 Q 矩阵？

*   **纠正**：Mask **不影响** Q 矩阵的生成。
*   Mask 作用于 $Q \times K^T$ 得到的 **Score 矩阵**上。在 Softmax 之前，将 Mask 位置的数值设为负无穷（$-\infty$），这样 Softmax 后的概率就为 0，从而“遮蔽”掉未来的信息。

Decoder 模块输出是概率吗？如何获得词呢？

*   Decoder 最终输出一个向量（维度为词表大小）。
*   经过 Softmax 层后，变成**概率分布**。
*   获得词的方法：可以是 **Argmax**（取概率最大的词），也可以是 **Sampling**（根据概率采样，如 Top-k, Top-p）。

首先提供 decoder 以一个特殊token，如：BOS，表示开始。（凶猛的肱二头好像发过一篇类似的文章）

输出一个向量，语料库中每个字/ subword 及对应一个概率（经过softmax），输出最高概率的字/ subword。

第二轮输入为第一轮的输出。最后进行到某一轮，输出概率最高的是停止token，这时输出就会停止，decoder 的输出长度和输入长度没有关系。

---

自回归解码器（AT）性能一般强于非自回归解码器（NAT）：因为NAT 是并行生成token，所以丢失了输出 token 之间的依赖建模能力。例如：
```
源句: "I love you"
合理译文A: "我 爱 你"
合理译文B: "我 非常 爱 你"
AT 逐步生成，能在上文约束下选择一致的路径。  
NAT 各位置独立采样，可能混合多种模态，产生： "我 非常 你"  （词从不同模态混入）
```




encoder 和 decoder 的交互模块是 cross attn。

交叉注意力输入的键和值为编码器的输出，查询为解码器中上一模块的输出。具体来说，decoder 输入为一个向量或特殊 token，然后经过多头自注意力掩码模块，得到一个 $q$，然后将经过encoder的输出的每个向量作为k和v，计算权重。

原始论文中的解码器的输入为编码器最后一层的输出，也存在其他的变体，参考论文：Rethinking and Improving Natural Language Generation withLayer-Wise Multi-View Decoding

![[Pasted image 20251210161713.png]]

## 训练

让 decoder 输出和 ground truch 的交叉熵越小越好。

teacher forcing：使用 ground truth 作为 decoder 的输入


## Transformer 相较于 RNN 的改进

1. 并行计算
2. 因为 attention 机制能一次获取全局信息，所以最长计算路径短
3. 可以捕捉长距离依赖关系

> 时间复杂度分析： https://ai.feishu.cn/wiki/XP00wxbvciWP1vk0qsicOESpnCg?from=from_parent_docx

| Layer Type                  | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --------------------------- | -------------------- | --------------------- | ------------------- |
| Self-Attention              | $O(n²·d)$            | $O(1)$                | $O(1)$              |
| Recurrent                   | $O(n·d²)$            | $O(n)$                | $O(n)$              |
| Convolutional               | $O(k·n·d²)$          | $O(1)$                | $O(log_k(n))$       |
| Self-Attention (restricted) | $O(r·n·d)$           | $O(1)$                | $O(n/r)$            |


---

## 数学基础

三角函数积化和差公式

$$
\begin{array} \\
sin(a+b)=sina \; cosb+cosa \; sinb \\
cos(a+b)​=cosa \; cosb−sina \; sinb
\end{array}
$$


---


参考资料：

[The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-transformer/) 该文章的中译版本：[细节拉满，全网最详细的Transformer介绍（含大量插图）！ - 知乎](https://zhuanlan.zhihu.com/p/681532180)、[Transformer 的训练与推理 | 暗香画楼](https://ww-rm.top/posts/2023/08/29/transformer/)

[Transformer常见问题与回答总结 - 知乎](https://zhuanlan.zhihu.com/p/496012402?utm_medium=social&utm_oi=629375409599549440)：用于检测是否理解了 Transformer 模型

[Encoder 模块的代码复现视频](https://www.bilibili.com/video/BV1cZ1xYFEFx)

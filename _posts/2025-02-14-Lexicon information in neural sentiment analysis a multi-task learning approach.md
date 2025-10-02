---
title: Lexicon information in neural sentiment analysis a multi-task learning approach
date: 2025-02-14 00:00:00 +0800 
categories: [论文] 
tags: [情感预测, 情感词典]    
math: true
---

阅读这篇论文的目的是了解这篇论文如何结合情感词典实现句子级别的情感预测，所以对论文中的「引言」、「文献综述」部分略读。

## 引言

研究问题：如何将外部知识（情感词典）结合到神经模型中，提高情感预测的性能

问题重要性：

- 结合外部知识（如情感词典）可以进一步提高模型的性能。
- 神经模型通常难以解释，而情感词典具有完全透明的优势，可以增强模型的可解释性。
- 情感词典易于适应和更新，可以更好地应对不同领域和数据的变化。

研究思路：采用多任务学习（multi-task learning, MTL）的架构实现同时处理两个任务。具体来说是，通过辅助任务学习情感词典这一外部知识，同时使用 BiLstm 情感分类器预测句子的情感类别。

文章的创新点或贡献：

- 多任务学习框架。在句子级别情感分类器中结合情感词典信息。
-  新挪威语词典。提供了一个新的挪威语情感词典，并且展示如何设计一个其他语言的情感词典
- 在英语和挪威语数据集上进行了实验，证明了模型的有效性和鲁棒性。

## 文献综述

从 4 个方面回顾相关工作：（i）情感词典，（ii）基于词典的情感分析方法，（iii）在神经模型中使用词典信息，（iv）NLP中的多任务学习。

（ii）基于词典的情感分析方法：根据形容词的情感分数之和->增加否定、加强、虚拟、减弱词对情感分数的分配+增加形容词、名、动、副词。基于专家手动构建的词典比使用少量种子词通过算法扩展词典更好；使用手动构建词典的模型面对各种领域的鲁棒性比机器学习模型更强。

## 模型的设计

### 模型框架图

![模型框架图](https://raw.githubusercontent.com/InTheFuture7/attachment/refs/heads/main/202502111652967.png)

### 框架描述

多任务模型共享下面两层。
#### 辅助任务

情感词典预测（辅助任务）用来预测单词是正面还是负面，从而优化 embedding layer 对句中单词的嵌入。为预测句子情感而增加单词情感嵌入，是因为认为单词情感预测对于句子情感预测具有较高的预测性（从实验结果表 3 分析得出）。

辅助任务的输入为使用预训练获得的单词嵌入，输出为情感类别（积极/消极）。

#### 主要任务

句子情感预测模型（主要任务）：将经过情感词典优化的词嵌入，输入 BiLSTM（为序列中每个词语提供一个综合其前后文信息的表示）中，输出不同情感类别的概率分布。

#### 初始词嵌入

- **英语**：使用来自GoogleNews的预训练300维词嵌入。
- **挪威语**：使用100维的skip-gram fastText词嵌入，这些嵌入是在NoWaC语料库上训练得到的（Bojanowski et al., 2016）。


## 实验设计

### 数据集

英文句子数据集：**斯坦福情感树库 SST**，从英文电影评论中提取的 11855 个句子。每个句子被打标签为：强烈负面、负面、中立、正面、强烈正面。

主要任务是句子级别的情感分类，辅助任务是单词级别的情感预测。

挪威文句子数据集：挪威评论语料库 NoReC 的 **NoReCeval**，来自不同领域的完整文本评论，包括了分布在 298 篇文档中的 7,961 个句子。标注：评价、事实暗示、非评价。

主要任务是句子级别的评价性分类，辅助任务是单词级别的情感预测。

### 情感词典

情感词典
* 英文情感词典：Hu 和 Liu
* 挪威文情感词典：基于 Hu 和 Liu 的，并作如下三种调整
	* translated：机器翻译+人工修正
	* full-forms：基于 translated+挪威语全形时词典 SCARRIE 扩展 translated 中单词的其他形式
	* lemmas：从 full-forms 中还原单词的基本形式。覆盖广而且精简 

![生成挪威文情感词典](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502111651397.png)

### 对比算法

- **LEXICON**: 使用情感词典的模型。利用情感词典为句子中的每个词赋予情感极性标签（+1 表示正向，-1 表示负向，0 表示中性），然后将这些标签作为特征输入到线性 SVM 中进行分类。
- **BOW**: Bag-of-Words（词袋模型）。使用词袋模型（Bag-of-Words）将句子表示为单词的集合，忽略单词的顺序，然后将这些词频信息作为特征输入到线性 SVM 中进行分类。
- **BOW+LEXICON**: 结合词袋模型和情感词典的模型。是在 `Bow` 模型的基础上增加了两个额外的特征。
- **LEX-EMB**: 使用词嵌入（Word Embeddings）的情感词典模型。使用情感词典学习词的情感嵌入，然后拼接原始词嵌入，得到带有情感信息的词表示，输入到 BiLstm 中，进行句子级别的情感分类。相较于 MTL 中交替训练（句子分类和词级情感预测）的区别是：Lex-Emb 学习情感词典中词嵌入和输入 BiLstm 为两个阶段，静态融合词典信息。
- **STL**: 单任务学习（Single Task Learning）模型。不进行辅助任务，将句子中每个词转为与训练的词嵌入向量，然后输入 BiLstm 中，训练情感极性。
- **MTL**: 多任务学习（Multi-Task Learning）模型。如图 1 所示。

### 评估指标 - Macro F1

Macro－F1 计算方式：

1. 计算第 $i$ 类的 Precision 和 Recall：
$$
\begin{gathered}
\operatorname{Precision}_i=\frac{\mathrm{TP}_i}{\mathrm{TP}_i+\mathrm{FP}_i} \\
\operatorname{Recall}_i=\frac{\mathrm{TP}_i}{\mathrm{TP}_i+\mathrm{FN}_i} .
\end{gathered}
$$

（1）对各类别的 Precision 和 Recall 求平均：

$$
\begin{aligned}
\text { Precision }_{\text {macro }} & =\frac{\sum_{i=1}^n \operatorname{Precision}_i}{n} \\
\text { Recall }_{\text {macro }} & =\frac{\sum_{i=1}^n \operatorname{Recall}_i}{n}
\end{aligned}
$$

（2）利用 F1 计算公式，即可得到Macro－F1。

$$
F 1_{\text {macro }}=2 \cdot \frac{\text { Precision }_{\text {macro }} \cdot \text { Recall }_{\text {macro }}}{\text { Precision }_{\text {macro }}+\text { Recall }_{\text {macro }}}
$$

选择 MacroF1 而非精度（accuracy）的原因：accuracy 易受多数类别的高准确率影响。

假设有一个二分类问题，数据集中有 $100$ 个样本，其中 $90$ 个样本属于多数类别（正类），$10$ 个样本属于少数类别（负类）。某个模型的预测结果如下：
- 多数类别（正类）：
	- 预测正确： 81 个（ $90\%$ ）
	- 预测错误：9 个（10％）
- 少数类别（负类）：
	- 预测正确：0 个（0％）
	- 预测错误：10 个（100％）

准确率的计算公式是：

$$
\text { Accuracy }=\frac{\text { 正确预测的样本数 }}{\text { 总样本数 }}=\frac{81 + 0}{ 100 } = 0.81
$$
### 模型训练

**交替训练**：在训练期间，一个 epoch 训练主要任务，下一个 epoch 训练辅助任务，交替训练的方式。（实验表明，每批之间的交替训练或从这两个任务中均匀抽样批次等复杂训练策略并没有改善指标效果）

**训练细节**：
- **优化器**：使用 Adam 优化器（Kingma and Ba, 2014）进行训练。
- **训练轮数**：模型训练 10 个 epochs，并采用早期停止策略，根据主任务在开发集上的表现来决定是否提前终止训练。
- **随机初始化敏感性**：由于神经网络模型对参数的随机初始化非常敏感，因此进行了五次独立的运行，每次运行使用不同的随机种子。最终结果展示的是这五次运行的平均值和标准差。
- **一致性**：为了确保不同模型之间的公平比较，所有实验都使用相同的五个随机种子。

每个模型运行 5 次，每次运行使用不同的随机种子，最终结果求 5 次运行的平均值。对于 MTL 模型而言，每次运行，一共有 10 个 epoch，辅助任务和主要任务交替 epoch。

### 实验结果

#### 主要任务对比

![](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502122135338.png)

1. Bow 在 SST 数据集上效果好，但在 NoReCeval 上的效果比 STL 相差较多
2. Bow+LEXICON 在两个数据集上都完胜 Bow，表明从情感词典学习到的特征有助于预测
3. LEX-EMB 在 SST 数据集上效果一般，但是在 NoReCeval 上效果较好
4. STL 在两个任务上比 BOW、LEX-EMB 都要好，而且在 NoReCeval 上都好
5. MTL 完胜其他模型

#### 辅助任务性能对比

下表为评估 MTL 和 LEX-EMB 模型在辅助词典预测任务上的性能。

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502130854209.png)

表 5 显示，LEX-EMB 模型在英语和挪威语上的 MTL 模型优于 MTL 模型。

假设差异是由于任务相似性引起的，而不是归因于语言的差异。对于英语而言，辅助任务更能预测主要任务（句子级别的情感），而对于挪威语，预测评估，事实暗示和非评估的主要任务并不十分取决于单词级别的性质。因此，挪威语中的 MTL 分类器较少依赖于辅助模块。

#### 错误分析

通过相对混淆矩阵，对比 MTL 和 STL 两种架构在各种标签上预测的数量差异。其中，深紫色表明 MTL 预测数量比 STL 要多，反之白色表明要少。

| SST 数据集预测数量差异                                                                                    | NoReCeval 数据集预测数量差异                                                                              |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| ![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502122217250.png) | ![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502122220875.png) |

在 SST 数据集上的预测数量差异表明，MTL 架构在正确预测情感极性强（StrongNeg、StrongPos）和中性的更多，但是对于一般极性的预测偏少。

在 NoReCeval 数据集上，MTL 架构在预测 Fact-implied、Non-evaluative 正确预测的数量多，在 Evaluate 上预测出来的数量少。


![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502131426460.png)

上表展示 MTL 比 STL 更好或更糟的样例。

Gold、STL、MTL 分别表示真实标签、STL 预测标签、MTL 预测标签。红色和蓝色框分别表示通过辅助任务训练得到的消极情感词和积极情感词。

#### 模型鲁棒性

![image.png](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202502130857733.png)

1. 使用词典有助于预测单词情感极性

2. MTL 模型使用词典，MacroF1 的得分普遍高于 STL 模型。表明 MTL 模型对不同情感信息来源的鲁棒性。关于鲁棒性的解释：模型在面对不同来源的情感信息时，仍然能够保持较好的性能。换句话说，MTL 模型对不同的情感词典都表现出了较好的适应能力，不会因为词典的不同而导致性能大幅波动。

3. 数据集的大小似乎比具体内容更重要，因为所有超过4000个词的词典都取得了类似的分数。

## 结论、改进和展望

### 结论

1. 使用英文和挪威文情感词典上做了词级的情感实验，本文方法相较于「单任务学习」有性能提升。

2. 多任务目标有助于中性和少数类（the multi-task objective tends to help the neutral and minority classes, indicating a regularizing effect.）

### 改进和展望

1. 模型忽视子词信息（如*unimpressive*）和多词表达（如*not my cup of tea*），如果能包括这些信息能有效提升结果

2. Although we have limited the scope of our auxiliary task to binary classification, using a regression task with sentiment and emotion labels may provide more fine-grained signal to the classifier. 二元分类任务 -> 带有情感及情绪标签的回归任务：判断情绪类别，并提供情绪的强烈程度（回归任务输出一个连续的数值，区分情感间的差异）

3. 增加方面级分类任务。"这部电影剧情精彩，但特效一般"涉及两个方面，"剧情"方面为积极；"特效"方面为消极

4. MTL 方法能将其他类型的外部知识引入神经分类器中，用于除情感分析以外的其他类型的任务。




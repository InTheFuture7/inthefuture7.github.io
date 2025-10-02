---
title: Explainable Image Similarity Integrating Siamese Networks and Grad-CAM
date: 2025-02-14 00:00:00 +0800 
categories: [论文] 
tags: [相似度计算, 可解释]    
---

> 给出论文（Explainable Image Similarity Integrating Siamese Networks and Grad-CAM）的内容解读、代码运行说明

论文链接：[J. Imaging | Free Full-Text | Explainable Image Similarity: Integrating Siamese Networks and Grad-CAM (mdpi.com)](https://www.mdpi.com/2313-433X/9/10/224)

代码文件：[ioannislivieris/Grad_CAM_Siamese (github.com)](https://github.com/ioannislivieris/Grad_CAM_Siamese)

## 论文理解

Grad CAM Siamese 算法集成了孪生网络和 Grad-CAM，以提供图像相似性任务的可解释性。前者用于计算两个输入图像之间的相似性，而后者用于可视化和解释由基于卷积的孪生网络做出的决策。

该算法的一个优点是，它能够提供图像相似度评分以及支持决策的视觉直观解释（事实解释）和不支持决策的解释（反事实解释）。

非事实性解释：当删除这些特征，决策会更加准确，或理解为：a description of “what would have not happened when a certain decision was taken”。详见参考文献19：Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization


### 数据集

1. **花卉数据集（Flowers dataset）**:
   - 包含4242张花卉图片，分辨率为320×240。
   - 图片被分类为五个类别：“chamomile”（甘菊）、“tulip”（郁金香）、“rose”（玫瑰）、“sunflower”（向日葵）和“dandelion”（蒲公英）。
2. **皮肤癌数据集（Skin cancer dataset）**:
   - 包含2800张图片，分辨率为224×224。
   - 图片涉及1400个恶性（Malignant）和1400个良性（Benign）的肿瘤病例。
3. **AirBnB数据集（AirBnB dataset）**:
   - 包含864张室内和室外房屋图片，分辨率为600×400。
   - 图片被分类为12个类别：“backyard”（后院）、“basement”（地下室）、“bathroom”（浴室）、“bedroom”（卧室）、“decor”（装饰）、“dining-room”（餐厅）、“entrance”（入口）、“house-exterior”（房屋外观）、“kitchen”（厨房）、“living-room”（客厅）、“outdoor”（户外）、“staircase”（楼梯）和“TV room”（电视房）。


### 模型训练过程

1. 创建训练对。对于训练数据集中的每张图片，研究者会随机选择另外两张图片。其中一张图片与原始图片属于同一个类别（同类图片），另一张图片属于不同的类别（异类图片）。也因此，Training instances、Validation instances、Testing instances输出均为偶数，而且三者和为输入图片数的两倍。
2. 分配标签。如果两张图片属于同一个类别，这对图片会被标记为标签零（0）。这意味着它们是相似的或相同的。如果两张图片属于不同的类别，这对图片会被标记为标签一（1）。这意味着它们是不同的。
3. 定义相似度。孪生网络的输出是一个距离值，它表示两张图片之间的相似程度。距离值越小，表示图片越相似；距离值越大，表示图片越不相似。相似度是通过 1 减去距离值 d 来定义的，即相似度 = 1 − d。这样，如果d接近0（即图片非常相似），相似度就会接近1；如果d较大（即图片不相似），相似度就会接近0。当然可以预设阈值为 0.5 来准确判断相似与否，这也是论文中所采用的方式。
4. 通过随机选择正样本（同一类别的图片）和负样本（不同类别的图片），孪生网络可以通过对比学习来优化其参数，使得正样本间的距离减小，负样本间的距离增大。
5. 对于正样本对（同一类别的图片），我们希望这个得分接近1（或较高的值），表示它们是相似的；对于负样本对（不同类别的图片），我们希望得分接近0（或较低的值），表示它们是不相似的。


关于计算相似度值的细节：

`02.Inference.py` 中 `pred = model(image1, image2)`，调用 `utils/Siamese.py` 的 `forward()`，进一步利用欧氏距离 `F.pairwise_distance` 计算距离值，并将这个距离视为不相似度，后面 1-model() 就是相似度。


损失函数计算公式
$$
\mathcal{L}=\frac{1}{2}\left[(1-y)\left(D_w\right)^2+y\left\{\max \left(0, m-D_w\right)\right\}^2\right],
$$
其中，$y$ 表示两张图片是否属于同一类别；$D_w$ 是两张图片的相似度值，可以采用；$m$ 为边际值，论文中被设定为 2。

损失函数的取值范围为：[0, +∞]，详见参考文献34：Understanding the Behaviour of Contrastive Loss


### 应用场景

论文针对三种数据集分别进行训练和评估，下面详细介绍花卉、房间数据集的结果，以便更好理解事实性解释和非事实性解释以及对模型输出的解读。

![image-20240603105636780](E:\InBox\Backup\image-typora-master\image-20240603105636780.png)

上面两个图象均属于玫瑰类别。模型的预测值为 0.24，也即模型预测输入图像间的相似性为 76%。由于相似度得分大于预定义的阈值 0.5，因此模型提示输入图像属于同一类。图 a、d 为原图，对比图 b、e，可知模型做出相似的的决策依据为花瓣处，对比图 c、f，可知模型做出不相似的的决策依据为花茎处，或者说，如果改变两朵花的花茎处，那么能提高图像的相似度，使得相似这一决策更加准确。


![image-20240603110645756](E:\InBox\Backup\image-typora-master\image-20240603110645756.png)

上面两个图象均属于 AirBnB 数据集中的不同类别，即第一张图片属于卧室类，而第二张图片属于客厅类。模型预测值为 0.516，即相似度得分为 48.4%，表明该模型预测输入图像大致属于不同的类。图 b、e 表明模型将重点放在第一张图像中的椅子和第二张图像中的灯、壁炉和时钟上，以预测图像不相似。图 c、f 给出了两幅图像的反事实解释，这表明模型分别关注了第一幅和第二幅图像中的床和沙发，以及两幅图像中呈现的桌子。两张图像都有一个共同的项目(桌子)以及两个视觉上相似的项目(床和沙发)，所以有 48.4% 的相似得分。

## 代码复现

### 代码层级目录

```
- checkpoints  # 存放训练好的模型
  - car
    |-- model.pth  # 此文件为未训练至拟合的模型
- Data  # AirBnB、Flowers、Skin_cancer 均为论文提供的数据集
        # data_for_compare 是用于计算相似度的图片文件夹
  - AirBnB
    |-- class 1  # 每个class文件夹下，存放图片文件
    |-- class 2
    |-- ...
  - car
    |-- class 1
    |-- class 2
    |-- ...
  - Flowers
    |-- class 1
    |-- class 2
    |-- ...
  - ...
- utils  # 工具类文件夹
  - dataset.py
  - early_stopping.py
  - Grad_CAM.py
  - imshow.py
  - ...
- 01.Train_Siamese_model.ipynb  # 用于训练模型，测试集评估模型
- 02.Inference.ipynb  # 利用训练好的模型计算两个图片的相似度
- environment.yml  # 环境依赖
- README.md
```



### 环境配置

#### conda 环境

基础环境：[Anaconda 3](https://www.anaconda.com/download/success)、文本编辑器（非必要，如：vscode、notepad++）

方式一：

```shell
# 切换路径至 environment.yml 所在目录
> conda env create -f environment.yml
# 切换虚拟环境
> conda activate pytorch_siamese
```

方式二：（推荐）

```shell
# 根据 environment.yml 文件中的 python 版本信息，先创建带有 Python 版本的虚拟环境（h6244533_0 标识特定版本）
> conda create -n pytorch_siamese python=3.8.17=h6244533_0
# 切换虚拟环境
> conda activate pytorch_siamese
# 若网络下载慢，则做如下设置。意为：若 1000s 没有收到任何数据，就认为是一个超时错误，默认值为 60s
> conda config --set remote_read_timeout_secs 1000.0
# 切换目录至 environment.yml 所在路径，安装其他包
> conda env update --file environment.yml
```

不需要设置镜像源，保持默认即可。

> 如果下载扩展包失败，可以尝试将 `environment.yml` 文件拆封成多个文件，多次小批量下载 Python 扩展包，定位到不易下载的扩展包，搜索相关解决办法。
>
> 添加镜像源下载 Python 扩展包时，可能出现报错信息有：error3、error4，因未能解决，故不推荐设置镜像源。

#### GPU 配置

`nvcc -V` 查看 CUDA 编译器版本；`nvidia-smi` 查看 NVIDIA 驱动支持 CUDA 的版本。

较低版本的 CUDA 编译版本可以在较高版本驱动下运行，为了确保最佳兼容性和充分利用新特性，理想的情况是确保 CUDA 工具包版本与 GPU 驱动支持的 CUDA 运行时版本相匹配或相近。

有时需要指定某一块 GPU 用于训练模型，需要为每块 GPU 指定唯一的编号，而后在代码中指定 GPU。



## 代码运行

1. 更新数据集。汽车图片数据存放在 `\Data\car` 中，图片大小与已有汽车图片保持统一，均为：宽 1200px，高 800px，若大小不一致，则 `02.Inference.ipynb` 输出的图像变形，暂时未知是否对模型训练有影响。子文件夹按汽车类别划分，文件夹名称无要求，最好直观可理解，目录层级关系详见「代码层级目录」。
2. 更新 `config.yml` 文件中的参数，如：backbone_model、optimizer。
3. 运行 `01.Train_Siamese_model.ipynb`，训练模型并用测试集评估。运行 Training 部分可能会报错 「报错记录及解决」部分的 error1，按提供的解决方法操作即可。
4. 运行 `02.Inferences.ipynb` 计算两个图像的相似度，并以热力图的形式展示两张图像的事实性解释（Factual explanations）和非事实性解释（Counterfactual explanations），以获得可解释的结果。其中，用于比对相似度的图像，需要手动设置路径。代码会输出热力图等图片到当前路径下，需及时存放输出图片，以免被覆盖。



## 报错记录及解决

### error1 - Initializing libiomp5md.dll, but found libiomp5md.dll already initialized

本报错信息，出现在 Training 部分。

详细报错信息：

```shell
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```

解决办法：

若代码基于 `conda` 虚拟环境 `pytorch_siamese` 中的 Python 运行，那么删除 `.\anaconda3\envs\pytorch_siamese\Library\bin` 中的 `libiomp5md.dll`。

若代码基于 base 环境中的 Python运行（不建议这样），则删除  `.\anaconda3\Library\bin\libiomp5md.dll` 文件。

参考链接：

[关于OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/371649016)

[就是他！让你的python内核莫名挂掉 | KMP_DUPLICATE_LIB_OK=TRUE 的始作俑者 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/655915099)



### error2 - CondaError: Downloaded bytes did not match Content-Length

```shell
CondaError: Downloaded bytes did not match Content-Length
  url: https://conda.anaconda.org/nvidia/win-64/libcublas-dev-11.11.3.6-0.tar.bz2
  target_path: D:\Program\anaconda3\pkgs\libcublas-dev-11.11.3.6-0.tar.bz2
  Content-Length: 394161490
  downloaded bytes: 389249421
```

```shell
> conda config --set remote_read_timeout_secs 1000.0
```



### error3 - CondaHTTPError: HTTP 429 TOO MANY REQUESTS 

```shell
CondaHTTPError: HTTP 429 TOO MANY REQUESTS for url <https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/win-64/repodata.json>
Elapsed: 00:49.357271

An HTTP error occurred when trying to retrieve this URL.
HTTP errors are often intermittent, and a simple retry will get you on your way.
'https//mirrors.ustc.edu.cn/anaconda/cloud/menpo/win-64'
```

不清楚是不是访问量太多，导致报错。尝试删除在镜像源中报错链接，但又会出现新的报错链接。



### error4 - 尝试 `aliyun` 报错

```shell
UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel anaconda/pkgs/msys2 <http://mirrors.aliyun.com/anaconda/pkgs/msys2>
```

应该是阿里云的链接无法访问。



### error5 - 根据 environment.yml 下载扩展包时，报错 UnicodeDecodeError

详细报错信息：`UnicodeDecodeError: 'gbk' codec can't decode byte 0xff in position 0: illegal multibyte sequence`

原因：yaml 文件编码非 utf-8，重新设置编码即可

具体操作（以 vscode 为例）：用 vscode 打开 yaml 文件后，点击右下角文件编码，`save with Encoding` -> `UTF-8`



### error6 - torch.cuda.OutOfMemoryError: CUDA out of memory

x详细报错信息：torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 236.00 MiB (GPU 0; 23.65 GiB total capacity; 13.64 GiB already allocated; 49.12 MiB free; 18.92 GiB allowed; 13.85 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation

解决办法：减小batch_size



### error7 - 下载包时，报错 `Solving environment: failed`、`ResolvePackageNotFound`

原因：`environment.yml` 中对扩展包除了有版本号约束，还有特定版本标识，如：` tqdm=4.65.0=pyhd8ed1ab_1`，这导致难以找到对应的版本。

解决方法1：将 `conda-forge` 添加到 channel 列表，以便搜索扩展包时也在 conda-forge channel 中查找：

```shell
> conda config --append channels conda-forge
```

解决方法2：根据报错信息的提示，进入 [Anaconda Cloud](https://anaconda.org) 中，寻找指定版本的扩展包并下载。

1. 搜索指定扩展包
2. 查看列表中的扩展包及版本号
3. 点击目标版本的扩展包链接
4. 在 `intallers` 部分，找到 conda 下载命令

解决方法3：删除 `yaml` 文件中的第二个等号后的内容。如果仍报错，则将报错的扩展包暂时不安装，等其他扩展包安装好后再安装。

解决方法4：在 [Anaconda Cloud](https://anaconda.org) 中寻找指定版本扩展包的 `Files`，离线下载扩展包。此方法不推荐，可能导致 conda 无法识别到来源（unknown），并报错。



### 补充：Linux 系统/服务器中安装 Anaconda

1. 通过在 `Linux` 终端中运行 `uname -m` 命令来确定的系统架构

2. 下载 Anaconda。访问 [Anaconda 官方网站](https://repo.anaconda.com/archive) 或者 [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 等提供下载链接的网站。

3. 选择版本。在下载页面，找到对应于的操作系统和系统架构的 Anaconda 安装包链接。

4. 使用 `wget` 命令在 Linux 终端中下载安装包。例如：

   ```shell
   wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
   ```

5. 下载完成后，给予安装脚本执行权限：

   ```shell
   chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
   ```

6. 执行安装脚本开始安装过程：

   ```shell
   ./Anaconda3-2024.02-1-Linux-x86_64.sh
   ```

7. 按照安装程序的提示进行操作，阅读并接受许可协议，选择安装路径，并决定是否要初始化 Anaconda 环境变量。浏览协议，enter，最后yes

8. 配置环境变量

   ```shell
   vim ~/.bashrc
   
   # 在末尾添加如下语句,此处路径为 anacodda3 实际安装路径
   export PATH=/home/xxxx/anacodnae/bin:$PATH
   
   # 添加完后激活环境
   source ~/bashrc
   ```

9. `conda -V` 测试



---



参考文章：

[PyTorch 代码中 GPU 编号与 nvidia-smi 命令中的 GPU 编号不一致问题解决方法](https://blog.csdn.net/sdnuwjw/article/details/111615052)

[PackagesNotFoundError: The following packages are not available from current channels的解决办法-CSDN博客](https://blog.csdn.net/weixin_45552562/article/details/109668589)

[解决创建conda环境时Solving environment: failed 和 ResolvePackageNotFound 的错误_solving environment: failed resolvepackagenotfound-CSDN博客](https://blog.csdn.net/hshudoudou/article/details/126407029)

[图像相似度分析——相似度算法 (qq.com)](https://mp.weixin.qq.com/s/FZQjwOu9MG-tfFxo6nyVsg)
[图片相似度计算(CVPR2015-DeepCompare) (qq.com)](https://mp.weixin.qq.com/s/dwyzaGv2k3MK2oUmVG5sUQ)
[无需训练/部署模型，一文学会图像相似度算法pHash原理和实战 (qq.com)](https://mp.weixin.qq.com/s/wj0Ap0fX72MrdfWFb3QfgA)
[教你如何实现图片特征向量提取与相似度计算 (qq.com)](https://mp.weixin.qq.com/s/EhehacVaiFUDmVaDRvWW9A)
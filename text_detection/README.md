# OCR深度实践系列：文本检测

**OCR深度实践系列：**

**（一）图像预处理**

**（二）数据生成**

目前OCR的研究集中在自然场景文本理解这一领域，其应用包括安全智能监控如车牌识别、智能穿戴设备应用如智能阅读、手机拍摄应用如关键字提取和智能搜索以及身份证/银行卡的关键信息识别。

本文是OCR深度实践系列的**第三篇**，主要介绍OCR的关键环节-文本检测，首先列出传统方法和深度学习方法，随后介绍深度学习的两种典型网络CTPN和CRAFT，最后给出这两种网络的实战演示。

**本文项目地址：https://github.com/Vincent131499/Chinese-OCR3/tree/master/text_detection**

场景文本检测的发展历程与绝大多数的计算机视觉任务相似，分为传统文本检测方法和基于深度学习的文本检测方法。

## 1.传统文本检测方法

传统的检测方法可分为两类：基于连通域的方法和基于滑动窗口的方法。

连通区域（Connected Component）一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域。

基于连通域的自然场景文本检测方法是通过提取图像中的连通区域获得文本候选区域，极大地缩小了搜索范围。然而这类方法大量依赖文本连通区域的检测结果，对文本检测召回率和文本轮廓的准确性的影响很大，所以对于该类方法，需要在保证文本连通区域检测高召回率的前提下，进一步思考如何获得准确的文本轮廓，从而提高文本检测的整体性能。然而在真实的场景中，由于光照变化、褪色、噪声干扰等因素，图像处理往往十分复杂，很难从中准确的检测出文本连通区域。

基于滑动窗口的文本检测方法是指在图像中滑动一个子窗口，同时在滑窗的各个位置应用检测算法。该类方法通常是基于单个字符的分类器，将滑动窗口作用于候选框，当场景很复杂时，比如光照、阴影的影响，导致字符分类稳定性变差，尤其是在不同场景下的检测。此外，文本行的排列较为随意，横的、竖的、斜的这些都会使得检测窗口的选取和后期文本行的生成难度增加。与此同时，如何选取合适的检测窗口滑动步长也是一个很繁琐的问题。

## 2.基于深度学习的文本检测方法

本篇重点介绍基于深度学习的文本检测方法。 在深度学习出现之前，场景文本检测的主要趋势是自下而上的，且大多数使用手工提取特征（例如MSER 或 SWT）作为基础的组件。近年来，通过采用流行的物体检测/分割方法，如SSD，Faster R-CNN和FCN，提出了基于深度学习的文本检测器。

该方法使用效果更佳稳定的高层语义特征，利用更多的数据去拟合更复杂、泛化能力更强的模型，在场景图片文本检测中取得了突破性的进展。具体可划分为两类方法：基于回归的方法、基于图像分割的方法。

### 2.1 基于回归的方法

针对基于回归的文本检测方法，其基本思路是先利用若干个Default Boxes（也称Anchor）产生大量的候选文本框，直接进行NMS后处理或者进一步精调再进行NMS后处理得到最终的检测结果。此处重点介绍CTPN网络模型，模型架构如下所示。

![CTPN-model](https://s3.ax1x.com/2020/11/18/DmxFjP.jpg)

具体包括7个步骤：

1）使用VGG16位backbone提取空间特征，取VGG的conv5层输出feature map；

2）使用3*3的滑窗针对feature map提取空间特征形成新的feature map；

3）将这个新的feature map进行reshape，输入到双向LSTM提取每一行的序列特征；

4）将双向LSTM的输出重新reshape，在经过一个FC卷积层;

5）经过类似Faster R-CNN的RPN网络获得text proposals；

6）使用NMS进行后处理，过滤多余的文本框；

7）假如理想的话（文本水平），会将上述得到的一个文本小框使用文本线构造方法合成一个完整文本行，如果还有些倾斜，会做一个矫正的操作。

连接文本提议网络CTPN(Connectionist Text Proposal Network)将文字沿文本行方向切割成更小且宽度固定的Proposal，极大的提高了检测定位的精度。同时，考虑到文本水平行的语义上下文信息，使用双向LSTM编码水平行的文本信息，进一步提高了网络的文本特征表示能力。为了解决文本任意方向排列的问题，引入方向信息将文本框划分为若干个Segment（可以是一个字符、一个单词，或几个字符）。

当然，CTPN也有一个很明显的缺点：对于非水平的文本的检测效果并不好。CTPN论文中给出的文本检测效果图都是文本位于水平方向的，显然CTPN并没有针对多方向的文本检测有深入的探讨。

### 2.2 基于图像分割的方法

对于基于图像分割的文本检测，其基本思路是通过分割网络结构进行像素级别的语义分割，进而基于分割的结果构建文本行。此处重点介绍CRAFT网络模型，模型架构如下图所示。

![CRAFT模型架构](https://s3.ax1x.com/2020/11/18/DmxEB8.png)

大多文本检测方法使用严格的word-level边界框进行训练，在表示任意形状的文本区域时会有所限制，而CRAFT则改进了这一点。CRAFT的网络结构如图。看起来并不复杂，基于VGG16的结构，整体类似UNet，是一个标准的分割模型，最终的输出有两个通道作为score map：Region Score 和 Affinity Score。Region Score表示该像素点是文字中心的概率，Affinity Score可以认为是该像素点是两个字符之间的中心的概率。这个结构还是比较简单的，其实大部分基于分割的模型网络结构都比较简单，主要是后处理与训练数据。

**训练数据生成：**

CRAFT的训练数据label不是二值化的图像，而是采用了类似热力图的图像，这也对应了上面说的，表示的是该位置在文字中心的概率。

![CRAFT-训练数据生成](https://s3.ax1x.com/2020/11/18/DmxeAg.png)

上图是训练数据的label的生成示意图。首先看左边，有了一个字符级的标注（第二个图的红框， Character Boxes），这个字符的四个点（第一个图绿边）构成一个四边形，做对角线，构成两个三角形（第一个图蓝边），取三角形的中心，两个框之间就有四个点，构成了一个新的边框，这个边框就是用来表示两个字符之间的连接的label的（Affinity Boxes）。第三个图是根据Box生成Label的过程，先生成一个2D的高斯图，通过透视变换，映射成与box相同的形状，最后粘到对应的背景上。

**后处理**

在网络输出score map之后，下面就要把这些像素级的label合成box，这篇论文里用的方法并不复杂，首先是通过阈值过滤score map，进行二值化，然后接一个连通域分析（Connected Component Labeling ），接下来通过连通域画出最终的QuadBox，可以看一下它的示意图：

![CRAFT-后处理](https://s3.ax1x.com/2020/11/18/DmxmNQ.png)

## 3.实战

本文以上面提及的CTPN和CRAFT为例针对输入图片进行文本检测。输入图片为：

![symbol](https://s3.ax1x.com/2020/11/18/DmxVHS.jpg)

### 3.1 CTPN演示

该项目在Chinese-OCR3/text_detection/ctpn_detection目录下。

Step1：构建所需的运行库

```bash
cd utils/bbox
chomod +x make.sh
./make.sh
```

Step2：下载预训练文件

百度云链接：链接：https://pan.baidu.com/s/1vMal539YjUr3EkLLUvhquw  提取码：l5sb 

Step3：将下载的预训练文件checkpoints_mlt放在ctpn_detection/目录下

Step4：运行demo

```bash
python ./main/demo.py
```

生成效果如下：

![ctpn_demo](https://s3.ax1x.com/2020/11/18/DmxAnf.jpg)

### 3.2 CRAFT演示

该项目在Chinese-OCR3/text_detection/craft_detection目录下。

分为两种方式使用：

**方式1：直接使用CRAFT的预训练模型测试自己的文本图像**

step1：下载CRAFT预训练权重文件craft_mlt_25k.pth(链接：https://pan.baidu.com/s/1sISrSV8Y-Zz7HxlPgPTtiA  提取码：svva)并将该权重文件放入pretrained目录下。 
step2：将需要检测的图像全部放入imgs目录下。 
step3：运行代码:

```bash
python test.py --trained_model ./pretrained/craft_mlt_25k.pth
```

step4：检测的结果将保存在result文件夹中供查看。 

生成效果如下：

![craft_demo](https://s3.ax1x.com/2020/11/18/DmxK9s.jpg)

**方式2：基于预训练模型在自己的数据集上继续训练，迁移学习**

step1：标注自己的数据集，使用标注工具**labelme**:，我们进行字符级别的标注，即对每个字符顺时针标注4个点构成一个多边形框，如下图所示：

![craft标注](https://s3.ax1x.com/2020/11/18/Dmxlj0.png)

然后我们给这个多边形框标注对应的字符，方便之后如果要做文本识别时使用。 

step2：假设我们数据集的根目录是blw，目录中有图片blw_1.jpg和标注blw_1.json两种文件，此时运行generate_score_map.py(注意修改main函数中的name = 你的根目录名称)，运行完之后，你的目录中除了上面两种.jpg和.json外，会多了blw_region_1.npy和blw_affinity_1.npy两种，分别对应了CRAFT中的region_map和affinity_map。

step3：将生成的四种文件分别放入data对应的子目录下

```bash
python file_construct.py
```

- 如下所示： 
  data：  affinity：blw_affinity_1.npy 
  anno：blw_1.json 
  img：blw_1.jpg 
  region：blw_region_1.npy 

 step4：此时我们自己的数据集就准备好了。 运行train.py(注意修改main函数中的参数设置)，训练好的模型默认存放在./models中。 

step5：使用训练好的模型进行文本检测

```bash
python test.py --trained_model ./models/5.pth
```

# 基于深度学习的OCR数据生成

基于深度学习的OCR系统一般分为文字检测和文字识别两个阶段，数据生成也需要针对这两个阶段分别叙述并辅以实战进行演示。

**本项目完整代码：https://github.com/Vincent131499/Chinese-OCR3/tree/master/data_generation**

## 1.文字检测数据的生成

### 1.1 SynthText方法

文字检测数据生成方法主要基于[Synthetic Data for Text Localisation in Natural Images](http://www.robots.ox.ac.uk/~ankush/textloc.pdf)提出的方法**SynthText**，介绍了如何生成自然场景下的文字图像。

**SynthText**方法的主要流程包括：

1）搜集业务相关的背景图片、文字语料和字体。其中背景图片是无文字的。

2）计算得到图片的语义与深度信息。论文代码中使用**gPb-UCM**方法得到图片的语义信息。在CV领域中的“语义信息”并不是指的是NLP中的上下文语义，而是各种语义区域；“深度信息”可以简单理解为图片与相机的距离。

3）获取符合条件的候选区域。具体操作分为两步：

step1：根据语义信息进行筛选，对每个分割片区进行遍历，利用OpenCV中的**minAreaRect**方法获取包含分割区内所有像素点的最小矩形区域。然后根据矩形的宽和高过滤掉宽高较小的区域。

step2：根据深度信息进行二次过滤，筛选出比较平整的区域。

4）对筛选出的候选区域进行图像变换，原图中的分割区域都是带有一定角度的，为了方便以后将单词或句子填充到相应的分割区域中，需要预先对每个分割区域做旋转变换。具体做法是：

先利用OpenCV的findContours()获取轮廓，将轮廓转换为3D形式，再将旋转后的区域平铺到平面上，对平面的区域进行旋转，使得minAreaRect()包围的矩形区域角度为0，随后利用OpenCV的findHomography()对旋转后分割区域的矩阵进行矩阵变换。

5）对变换后的区域进行填充。随机选择字体、文字内容、添加特效等，生成相应的文字图片，然后复制到相应的区域中。

### 1.2 实战演示

该项目位于Chinese-OCR3/data_generation/SynthText目录下。

**一般而言：开源的数据集已经足够用于文字检测项目，所以生成更多应用在文字识别阶段，这里只是作为演示使用。**

此处给出**SynthText 自然场景图像数据集(地址还未上，等下载完放到网盘)**，由80万个图像组成，大约有 800 万个合成单词实例。 每个文本实例都使用其文本字符串，字级和字符级边界框进行注释。

**安装依赖：**

```bash
pip install -r requirements.txt
```

**生成数据：**

```bash
python gen.py --viz
```

  - **dset.h5**: 里面有5张图片，可以下载其他图片
  - **data/fonts**: 一些字体
  - **data/newsgroup**: 一些语料
  - **data/models/colors_new.cp**: Color模型
  - **data/models**:模型相关
  - 生成的结果在results目录下

**可视化预览生成结果：**

```bash
python visualize_results.py
```

以下放出一张示例的生成图片：

![image-20201110152151504](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201110152151504.png)



## 2.文字识别数据的生成

深度学习系统中，在检测出目标之后，往往还需要使用分类器对检测区域进行识别。深度学习依赖大量的数据才能得到令人满意的识别效果。

在实际的业务场景中，首先需要根据具体的业务分析需要的背景、字体、颜色、形变以及语料等信息。具体识别数据的生成流程如下所示：

![识别数据生成流程图](F:\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\绘图\识别数据生成流程图.jpg)

目前常用流行的识别数据生成方法可大致分为三类：GAN生成法、基于特征变换的图像增强、基于深度学习的图像增强。

### 2.1 基于GAN生成数据

在很多场景下，真实数据往往非常稀缺和敏感，例如身份证数据、银行卡数据、车牌数据这些涉及个人信息的数据往往很难获取，而且很容易违反法律规定。借助GAN（Generative Adversarial Network，生成对抗网络）可以在一定程度上缓解上述问题。目前GAN的应用场景基本上覆盖了AI的所有领域，例如图像和音频的生成、图像风格迁移、图像修复（去噪和去马赛克）、NLP中的文本生成等。

生成对抗网络，顾名思义，就是在生成模型的基础上引入对抗博弈的思想。假设我们有一个图像生成模型Generator，它的目标是生成一张比较真实的图像，与此同时，我们还有一个图像判别模型Discriminator，它的目标是正确的判别一张图像是生成的还是真实的。具体流程如下：

- 1）生成模型Generator生成一批图像。
- 2）判别模型Discriminator学习区分生成图像和真实图像。
- 3）生成模型根据判别模型反馈结果来改进生成模型，迭代生成新图像。
- 4）判别模型继续学习区分生成图像和真实图像。

直到二者收敛，此时生成模型和判别模型都能达到比较好的效果。上述的博弈类似《射雕英雄传》中周伯通的左右互搏术，能循环提升生成模型和判别模型的能力。另外，在生成模型中采用神经网络作为主干/backbone，则称之为生成对抗网络。GAN模型结构如下图所示。

![GAN模型结构](F:\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\绘图\GAN模型结构.jpg)

在这里以改进的**pix2pix**经典模型为例进行实战演示。

此项目位于Chinese-OCR3/data_generation/pytorch-CycleGAN-and-pix2pix目录下。这里使用在facades数据集预训练好的pix2pix模型进行演示。

具体分为3步：

step1：下载预训练模型

```bash
bash ./scripts/download_pix2pix_model.shfacades_label2photo
```

step2：下载facades数据集

```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```

step3：生成结果

```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```

生成图片如下示例：

![pix2pix_demo1](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\绘图\pix2pix_demo1.png)

![pix2pix_demo2](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\绘图\pix2pix_demo2.png)

### 2.2 基于特征变换的图像增强

这类方法是对现有的数据进行**图像增广**进而扩充数据量。在文字识别的训练中，由于文字的特殊性，能够选择的增强方法有限，主要有以下4种类型：

- 1）模糊。
- 2）对比度变化。
- 3）拉伸。
- 4）旋转。

在这里分别针对这4中手段进行实战演示，该项目位于Chinese-OCR3/data_generation/augment目录下。

输入图片：

![sample](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\sample.png)

核心代码如下：

```python
#旋转
def rotate(img, angle, center=None, scale=1.0):
    # get the dimension of the img
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(
        img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return rotated_img

#拉伸（放大-缩小）
def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    # get the dimension of the img

    dm = None

    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width:
        r = width / float(w)
        dm = (width, int(h * r))
    else:
        r = height / float(h)
        dm = (int(w * r), height)

    resized_img = cv2.resize(img, dm, interpolation=inter)

    return resized_img

# 对比度变化
def adjust_brightness_contrast(img, brightness=0., contrast=0.):
    """
    Adjust the brightness or contrast of image
    """
    beta = 0
    return cv2.addWeighted(img, 1 + float(contrast) / 100., img, beta, float(brightness))

# 模糊
def blur(img, typ="gaussian", kernal=(2, 2)):
    """
    Blur the image
    :params:
            typ: "gaussian" or "median"
    """
    if typ == "gaussian":
        return cv2.GaussianBlur(img, kernal, 0, None, 0)
    elif typ == "median":
        return cv2.blur(img, kernal)
    else:
        return img

```

效果如下所示：

旋转-倾斜一定角度：

![roated_img](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\roated_img.png)

拉伸-放大：

![resized_long_img](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\resized_long_img.png)

拉伸-缩小：

![resized_short_img](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\resized_short_img.png)

对比度-增强：

![incre_contrasted](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\incre_contrasted.png)

对比度-降低：

![decre_contrasted](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\decre_contrasted.png)

模糊：

![blured](E:\my_code\算法平台研发\OCR组件研发\基于深度学习的文字识别教程\chinese-ocr3-back\data_generation\augment\blured.png)

### 2.3 基于深度学习的图像增强

这类方法也是对现有的数据进行**图像增广**进而扩充数据量。深度学习方法是“Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition”这篇论文提出的合成自然场景文本的方法，适用于文字识别。具体的文本生成过程分为六步：

1）字体渲染。

2）描边、加阴影、着色。

3）基础着色。

4）仿射投影扭曲。模拟3D环境。

5）自然数据混合。

6）加噪声。

此处给出两个资源：

1）Imgaug：https://github.com/aleju/imgaug   主要用于物体检测的增强。

2）Augmentor：https://github.com/mdbloice/Augmentor  做一些更复杂的仿射扭曲变换。




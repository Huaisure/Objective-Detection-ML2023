## 目标检测模型

本项目使用Faster R-CNN深度学习模型，Faster R-CNN 的主要组成部分包括：
1. 骨干网络（Backbone Network）：
    - 骨干网络通常是预训练的卷积神经网络（如 VGG 或 ResNet），用于提取图像的特征。
    - 图像通过这个网络，生成一组特征图（feature maps）。
2. 区域建议网络（Region Proposal Network, RPN）：
    - RPN 使用骨干网络的特征图来识别潜在的物体区域。
    - 它在特征图上滑动一个小窗口，并对每个位置输出一组区域建议（proposals），每个建议包含边界框坐标和物体存在的分数。
3. RoI 池化（Region of Interest Pooling）：
    - 对于每个区域建议，RoI 池化层提取固定大小的特征图（例如 7x7）。
    - 这一步骤确保不同大小的建议区域可以被处理成统一大小的输出，以便进行后续的分类和边界框回归。
4. 分类和边界框回归：
    - 对于每个 RoI 池化后的特征图，模型使用全连接层来预测物体的类别和调整边界框的坐标。
    - 这样，模型不仅可以识别图像中的物体，还可以精确地定位它们。   

Faster R-CNN 的主要优势在于它能够实现端到端的训练和较高的检测速度，同时保持较好的检测精度。它通过 RPN 有效地降低了候选区域的数量，并能精确地定位和分类物体。

本项目在`src/model.py`中创建了一个Faster R-CNN模型，本身未经过训练。使用经过预训练的ResNet50作为骨干网络。定义RPN（区域建议网络）的锚点生成器，以及感兴趣区域（ROI）的池化层，结合这些创建了Faster R-CNN模型。
```
model = FasterRCNN(
    backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    min_size=500,
)
```

## 数据预处理

数据预处理的各种功能函数位于`src/data_preprocessing.py`中。

### 主要功能函数

具体来说，调用的接口为`get_data_loaders()`函数，为用于准备和获取训练和验证数据集的数据加载器。主要功能包括数据集的划分、数据集的创建和数据加载器的初始化。

使用自己定义的`CustomVOCDataset`类创建训练和验证数据集，首先实现数据集的分割，通过自己定义的`parse_voc_xml()`函数读取分析xml文件，得到target的全部信息。

应用数据增强技术，具体的数据增强代码位于`src/utils.py`中的`train_transforms()`函数，下面详细讲讲这个函数中应用了哪些数据增强的方法。

### 数据增强

本项目中应用的数据增强技术如下：

#### 1. 随机旋转：

- 使用 torchvision.transforms.RandomRotation.get_params([-10, 10]) 获取 -10 到 10 度之间的随机旋转角度。
- 使用 F.rotate 函数将图像旋转这个角度。
- rotate_box 函数调整目标边界框的位置，以匹配旋转后的图像。

#### 2. 随机裁剪：

- 将图片随机裁剪为256*256的块，保证其中至少包含一个目标，相应的调整目标框。
- 在实验中发现效果并不好，于是放弃了这种方法。

#### 3. 随机调整亮度、对比度和饱和度：

- 使用 F.adjust_brightness、F.adjust_contrast 和 F.adjust_saturation 函数随机调整图像的亮度、对比度和饱和度。
- 这些调整通过 random.uniform(0.5, 1.5) 生成的随机因子来控制，增加了图像的变化性。

#### 4. 随机翻转：

- 以 50% 的概率对图像进行水平翻转。
- 如果翻转发生，相应地调整目标边界框的位置。

#### 5. 随机剪切：

- 在图像中随机生成一些“剪切”区域，这些区域的像素被置为0。
- 这有助于模型学习关注图像的不同部分，提高对部分遮挡物体的识别能力。

这些数据增强技术的使用有利有弊，许多参数需要在实际中调整，才能对模型的训练有所帮助。
## 训练

模型的训练脚本位于`src/train.py`中包含了整个训练过程的各个步骤，包括数据加载、模型初始化、训练循环、验证、早期停止判断以及结果保存。

### 调用示例

在根目录下开启终端，
```
python src/train.py --verbose
```
即可开始训练自己的模型，模型会保存在当前目录下；如果想导入自己的模型，可以通过
```
--load_model Your_Path_to_Model
```
来导入。

### 训练结果评估

使用测试集的mAP指标作为训练结果的评估，在本项目中调用pycocotools库完成mAP的计算。在得到预测的类别和目标框后，需要将数据转换为COCO格式后在进行计算。  

mAP（mean Average Precision）是一种在计算机视觉领域，尤其是在目标检测任务中广泛使用的评估指标。mAP 提供了一个整体的性能度量，反映了模型在多个类别和不同召回率（recall）下的精确度（precision）。这个指标特别重要，因为它不仅考虑了模型检测物体的准确性，还考虑了其对不同物体类别的检测能力。  

具体代码位于`src/utils.py`中的`validate`函数。

## 调用模型的接口

调用`src/detection.py`，提供待检测图片的路径或所在文件夹路径，即可检测其中的图片，并保存在指定路径下。

### 示例
可以在根目录下开启终端输入下面指令：
```
python src/detection.py \
--path_to_image Your_Path_to_Image \
--path_to_model Your_Path_to_Model \
--path_to_save Your_Path_to_Save
```

:) 如有疑惑、问题请联系我 at 328333607@qq.com
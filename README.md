## 目标检测模型

该模型使用torch的预训练模型faster_rcnn，数据集为PASCAL VOC2012数据集的子集（共1714张图片），其中2007_000559至2012_001051为训练集，其余为验证集。

- 预测输入图片中的物体位置（框）和类别
- 评价指标：验证集的mAP

---

## **Quick Start**

- **./data**: $~$ 存储训练数据，为图片和对应的xml文件
- **./src**: $~~~~$ 代码部分
    + data_preprocessing.py：处理数据；
    + model.py：模型文件；
    + detection.py: 调用模型进行测试，以及实际使用；
    + train.py：训练脚本，--load_model表示导入先前训练的模型继续训练： 
    
    ```
    python src/train.py --load_model --verbose
    ```
    + utils.py：包含一些功能函数

使用举例：（在根目录下）
```
python src/detection.py \
--path_to_image Your_Path_to_Image \
--path_to_model Your_Path_to_Model
```
可以完成对图片的目标检测，检测结果会以图片的形式存储在当前目录下。

⚠️ 如有疑问、建议请联系我 at 328333607@qq.com

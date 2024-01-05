"""
Description: This file contains the model for the application.
Author: Huaishuo Liu
Maintainer: Huaishuo Liu
Created: 2023-12-20
"""
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights


def create_faster_rcnn_model(num_classes):
    # 加载预训练的 ResNet50 模型作为 backbone
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=ResNet50_Weights.IMAGENET1K_V1
    )

    # RPN（Region Proposal Network）锚点生成器
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128, 256),) * 5, aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # ROI（Region of Interest）池化层
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    # 创建 Faster R-CNN 模型
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=500,
    )

    return model

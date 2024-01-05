import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
import random
from sklearn.metrics import average_precision_score
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


def transform_image_and_boxes(
    image, target, new_size=(256, 256)
):  # TODO: better new_size
    """
    transform the image and bounding boxes
    return: transformed image and bounding boxes
    """
    # 原始图像尺寸
    orig_size = torch.tensor(
        [image.width, image.height, image.width, image.height]
    ).unsqueeze(0)

    # 调整图像尺寸
    image = F.resize(image, new_size)

    # 调整边界框尺寸
    if "boxes" in target:
        # 计算缩放比例
        scale = torch.tensor(
            [new_size[1], new_size[0], new_size[1], new_size[0]]
        ).unsqueeze(0)
        target["boxes"] = (target["boxes"] / orig_size) * scale

    return image, target


def validate(model, data_loader, device, num_class):
    model.eval()
    coco_gt = COCO()
    coco_gt.dataset = {"images": [], "annotations": [], "categories": []}
    image_id = 0
    annotation_id = 0

    predictions = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, output in enumerate(outputs):
            # 添加图像信息
            coco_gt.dataset["images"].append({"id": image_id})

            # 添加真实标签
            for label, target in zip(targets[i]["labels"], targets[i]["boxes"]):
                x_min, y_min, x_max, y_max = target.unbind(0)
                width = x_max - x_min
                height = y_max - y_min
                coco_gt.dataset["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [
                            x_min.item(),
                            y_min.item(),
                            width.item(),
                            height.item(),
                        ],
                        "area": width.item() * height.item(),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            # 添加预测
            for box, label, score in zip(
                output["boxes"], output["labels"], output["scores"]
            ):
                x_min, y_min, x_max, y_max = box.unbind(0)
                width = x_max - x_min
                height = y_max - y_min
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": label.item(),
                        "bbox": [
                            x_min.item(),
                            y_min.item(),
                            width.item(),
                            height.item(),
                        ],
                        "score": score.item(),
                    }
                )

            image_id += 1

    # 添加类别信息
    for i in range(1, num_class):
        coco_gt.dataset["categories"].append({"id": i, "name": str(i)})

    coco_gt.createIndex()

    # 加载预测数据
    coco_dt = coco_gt.loadRes(predictions)

    # COCO 评估
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # 返回 mAP


def conditional_random_crop(img, targets, output_size=(256, 256)):
    """
    条件性随机裁剪图像，尽量确保至少包含一个目标。
    如果目标大于输出尺寸，则直接调整图像大小。

    Args:
    - img (PIL Image): 要裁剪的图像。
    - targets (dict): 包含 'boxes' 键的字典，其中包含边界框的坐标。
    - output_size (tuple): 裁剪后图像的尺寸 (width, height)。

    Returns:
    - PIL Image: 裁剪并调整大小后的图像。
    - dict: 更新后的目标字典。
    """
    boxes = targets["boxes"]
    w, h = img.size
    th, tw = output_size

    # 如果图片尺寸小于输出尺寸，则直接调整大小
    # 如果没有边界框或所有目标都大于输出尺寸，则直接调整大小
    if (
        len(boxes) == 0
        or (all(box[2] - box[0] > tw and box[3] - box[1] > th for box in boxes))
        or w < tw
        or h < th
    ):
        resized_img = F.resize(img, output_size)

        # 调整目标的边界框坐标
        scale_x, scale_y = tw / w, th / h
        new_boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        new_targets = targets.copy()
        new_targets["boxes"] = new_boxes.int()

        return resized_img, new_targets

    # 尝试随机选择包含至少一个目标的裁剪区域
    max_attempts = 10
    for _ in range(max_attempts):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        crop_box = [x1, y1, x1 + tw, y1 + th]

        # 检查是否至少有一个目标在裁剪区域内
        if any(
            (
                box[0] < crop_box[2]
                and box[2] > crop_box[0]
                and box[1] < crop_box[3]
                and box[3] > crop_box[1]
            )
            for box in boxes
        ):
            cropped_img = img.crop(crop_box)
            # 调整目标的边界框坐标
            new_boxes = boxes - torch.tensor([x1, y1, x1, y1])
            new_boxes = torch.clamp(new_boxes, min=0, max=tw)
            new_targets = targets.copy()
            new_targets["boxes"] = new_boxes

            # 确保边框的宽和高为正数，删去无效的边框和对应的标签
            new_boxes = new_targets["boxes"]
            # print(new_boxes)
            keep = (new_boxes[:, 2] > new_boxes[:, 0]) & (
                new_boxes[:, 3] > new_boxes[:, 1]
            )
            # print(keep)
            new_targets["boxes"] = new_boxes[keep]
            # print(new_targets["boxes"])
            new_targets["labels"] = new_targets["labels"][keep]
            # print(new_targets["labels"])

            return cropped_img, new_targets

    # 如果找不到合适的裁剪区域，直接调整大小
    resized_img = F.resize(img, output_size)

    # 调整目标的边界框坐标
    scale_x, scale_y = tw / w, th / h
    new_boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
    new_targets = targets.copy()
    new_targets["boxes"] = new_boxes

    return resized_img, new_targets


def rotate_box(bbox, angle, img_size):
    """
    Rotate the bounding box.

    Args:
    - bbox (Tensor): The bounding box in [x_min, y_min, x_max, y_max] format.
    - angle (float): The rotation angle in degrees.
    - img_size (tuple): The size of the image as (width, height).

    Returns:
    - Tensor: Rotated bounding box.
    """
    cx, cy = img_size[0] // 2, img_size[1] // 2  # Image center
    angle_rad = np.radians(angle)  # Convert angle to radians

    new_boxes = []
    for box in bbox:
        # Original coordinates
        x_min, y_min, x_max, y_max = box
        corners = np.array(
            [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
        )

        # Rotate corners
        corners = corners - np.array([cx, cy])
        corners = np.dot(
            corners,
            np.array(
                [
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)],
                ]
            ),
        )
        corners = corners + np.array([cx, cy])

        # Get new bounding box
        # 使边界框为最接近的整数
        corners = np.rint(corners).astype(np.int32)
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)

        new_box = torch.tensor([x_min, y_min, x_max, y_max])
        new_boxes.append(new_box.clamp(min=0))  # Ensure the bbox is within image bounds

    return torch.stack(new_boxes)


def random_cutout(img, probability=0.5, max_holes=1, max_length=40):
    """
    随机剪切函数。
    :param img: 输入图像。
    :param probability: 应用剪切的概率。
    :param max_holes: 最大剪切区域数量。
    :param max_length: 剪切区域的最大长度。
    :return: 处理后的图像。
    """
    if random.random() < probability:
        h, w = img.size(1), img.size(2)
        mask = torch.ones_like(img)
        for _ in range(random.randint(1, max_holes)):
            y = random.randint(0, h)
            x = random.randint(0, w)
            y1 = np.clip(y - max_length // 2, 0, h)
            y2 = np.clip(y + max_length // 2, 0, h)
            x1 = np.clip(x - max_length // 2, 0, w)
            x2 = np.clip(x + max_length // 2, 0, w)
            mask[:, y1:y2, x1:x2] = 0.0
        img = img * mask
    return img


def train_transforms(img, target):
    # 随机旋转
    angle = torchvision.transforms.RandomRotation.get_params([-10, 10])
    img = F.rotate(img, angle)
    target["boxes"] = rotate_box(target["boxes"], angle, img.size)

    # 随机裁剪
    # img, target = conditional_random_crop(img, target, output_size=(256, 256))
    # 经测试，随机裁剪会导致 mAP 下降

    # 随机调整亮度、对比度和饱和度
    img = F.adjust_brightness(img, brightness_factor=random.uniform(0.5, 1.5))
    img = F.adjust_contrast(img, contrast_factor=random.uniform(0.5, 1.5))
    img = F.adjust_saturation(img, saturation_factor=random.uniform(0.5, 1.5))

    # 随机翻转
    if np.random.rand() > 0.5:
        img = F.hflip(img)
        target["boxes"][:, [0, 2]] = img.size[0] - target["boxes"][:, [2, 0]]

    # 调整图像大小
    img, target = transform_image_and_boxes(img, target, new_size=(500, 500))

    img = F.to_tensor(img)
    # random cutout
    img = random_cutout(img, probability=0.5, max_holes=1, max_length=40)

    return img, target


def val_transforms(img, target):
    # 调整图像大小
    img, target = transform_image_and_boxes(img, target, new_size=(500, 500))
    img = F.to_tensor(img)
    return img, target

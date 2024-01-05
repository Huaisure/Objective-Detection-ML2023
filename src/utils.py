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


def calculate_ap_per_class(
    num_class,
    true_boxes,
    pred_boxes,
    pred_scores,
    pred_labels,
    true_labels,
    iou_threshold=0.5,
):
    # 按类别分类的预测边界框和分数
    pred_boxes_class = {i: [] for i in range(num_class)}
    true_boxes_class = {i: [] for i in range(num_class)}
    pred_scores_class = {i: [] for i in range(num_class)}

    # 将预测和真实边界框按类别分类
    for i in range(len(pred_boxes)):
        for p_box, p_score, p_label in zip(
            pred_boxes[i], pred_scores[i], pred_labels[i]
        ):
            p_label = p_label.item()
            pred_boxes_class[p_label].append(p_box)
            pred_scores_class[p_label].append(p_score)

        for t_box, t_label in zip(true_boxes[i][0], true_boxes[i][1]):
            t_label = t_label.item()
            true_boxes_class[t_label].append(t_box)

    # 对每个类别计算 AP
    aps = []
    for c in range(num_class):
        # 提取特定类别的预测框、分数和真实框
        c_pred_boxes = pred_boxes_class[c]
        c_true_boxes = true_boxes_class[c]
        c_pred_scores = pred_scores_class[c]

        # 检查预测框和真实框是否为空
        if len(c_pred_boxes) == 0 or len(c_true_boxes) == 0:
            continue  # 如果没有预测框或真实框，则跳过此类别

        # 按分数降序排序预测框的索引
        sorted_indices = np.argsort(-np.array(c_pred_scores))

        tp = np.zeros(len(c_pred_boxes))
        fp = np.zeros(len(c_pred_boxes))

        for i, idx in enumerate(sorted_indices):
            pred_box = c_pred_boxes[idx]

            # 仅当存在真实框时计算 IOU
            if len(c_true_boxes) > 0:
                iou_max = (
                    box_iou(torch.stack([pred_box]), torch.stack(c_true_boxes))
                    .max()
                    .item()
                )
            else:
                iou_max = 0

            if iou_max >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1

        # 计算累积 TP 和 FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(c_true_boxes)

        # 计算 AP
        ap = np.trapz(precision, recall)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


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


def calculate_aps(all_preds, all_targets, num_class):
    aps = []

    for class_id in range(1, num_class):  # 排除背景类别
        true_labels = []
        pred_scores = []

        for target, pred in zip(all_targets, all_preds):
            # 对于每个图像，检查是否存在当前类别的目标
            mask = target["labels"] == class_id
            true_labels.extend(mask.int().tolist())

            # 如果有预测分数，添加到列表中
            pred_mask = pred["labels"] == class_id
            if pred_mask.any():
                pred_scores.extend(pred["scores"][pred_mask].tolist())
            else:
                # 如果没有预测，添加一个低置信度值
                pred_scores.append(0.0)

        # 计算当前类别的 AP
        if true_labels and pred_scores:  # 确保列表不为空
            ap = average_precision_score(true_labels, pred_scores)
            aps.append(ap)

    return aps


def get_crop_params(img, target, output_size):
    """
    保证裁剪后仍有目标在图像中"""
    # 获取图像的宽度和高度
    w, h = img.size
    ischanged = False

    # 获取所有目标的边界框
    boxes = target["boxes"]

    # 如果没有目标，返回整个图像的参数
    if boxes.shape[0] == 0:
        return 0, 0, w, h, ischanged

    # 计算一个包含所有目标的边界框
    min_x = boxes[:, 0].min().item()
    min_y = boxes[:, 1].min().item()
    max_x = boxes[:, 2].max().item()
    max_y = boxes[:, 3].max().item()

    # 如果单个目标太大无法完全包含在裁剪区域内，返回整个图像的参数
    for box in boxes:
        if box[2] - box[0] > output_size[0] or box[3] - box[1] > output_size[1]:
            return 0, 0, w, h, ischanged

    # 确保裁剪区域至少包含一个目标
    i = random.randint(min_y, max_y - output_size[1])
    j = random.randint(min_x, max_x - output_size[0])
    ischanged = True

    return i, j, output_size[0], output_size[1], ischanged


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
    img, target = transform_image_and_boxes(img, target, new_size=(800, 800))

    img = F.to_tensor(img)
    # random cutout
    img = random_cutout(img, probability=0.5, max_holes=1, max_length=40)

    return img, target


def val_transforms(img, target):
    # 调整图像大小
    img, target = transform_image_and_boxes(img, target, new_size=(800, 800))
    img = F.to_tensor(img)
    return img, target

import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.ops import box_iou


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
    true_boxes, pred_boxes, pred_scores, pred_labels, true_labels = [], [], [], [], []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes.append(output["boxes"].cpu())
                pred_scores.append(output["scores"].cpu())
                pred_labels.append(output["labels"].cpu())

                true_boxes.append(
                    (targets[i]["boxes"].cpu(), targets[i]["labels"].cpu())
                )
                true_labels.extend(targets[i]["labels"].cpu().tolist())

    # 计算每个类别的 AP
    mAP = calculate_ap_per_class(
        num_class, true_boxes, pred_boxes, pred_scores, pred_labels, true_labels
    )
    return mAP


def train_transforms(img, target):
    # 添加数据增强步骤
    img = F.to_tensor(img)
    return img, target


def val_transforms(img, target):
    img = F.to_tensor(img)
    return img, target

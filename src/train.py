"""
Discription: train the model
Author: Huaishuo Liu
Maintainer: Huaishuo Liu
Created: 2023-12-20
"""
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from model import create_faster_rcnn_model
from data_preprocessing import get_data_loaders
from torchvision.ops import box_iou
import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_ap_per_class(num_class, true_boxes, pred_boxes, pred_scores, pred_labels, true_labels, iou_threshold=0.5):
    # 按类别分类的预测边界框和分数
    pred_boxes_class = {i: [] for i in range(num_class)}
    true_boxes_class = {i: [] for i in range(num_class)}
    pred_scores_class = {i: [] for i in range(num_class)}

    # 将预测和真实边界框按类别分类
    for i in range(len(pred_boxes)):
        for p_box, p_score, p_label in zip(pred_boxes[i], pred_scores[i], pred_labels[i]):
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
                iou_max = box_iou(torch.stack([pred_box]), torch.stack(c_true_boxes)).max().item()
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
                pred_boxes.append(output['boxes'].cpu())
                pred_scores.append(output['scores'].cpu())
                pred_labels.append(output['labels'].cpu())

                true_boxes.append((targets[i]['boxes'].cpu(), targets[i]['labels'].cpu()))
                true_labels.extend(targets[i]['labels'].cpu().tolist())

    # 计算每个类别的 AP
    mAP = calculate_ap_per_class(num_class, true_boxes, pred_boxes, pred_scores, pred_labels, true_labels)
    return mAP



def train_transforms(img, target):
    # 添加数据增强步骤
    img = F.to_tensor(img)
    return img, target


def val_transforms(img, target):
    img = F.to_tensor(img)
    return img, target


train_loader, val_loader = get_data_loaders(
    "./data", train_transforms, val_transforms, batch_size=4
)

num_classes = 21  # 20 类 + 1 背景类
model = create_faster_rcnn_model(num_classes).to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    # 训练过程显示，通过tqdm
    print("##########")
    print("Epoch:", epoch + 1)
    print("##########")
    for images, targets in tqdm.tqdm(train_loader):
        # print(targets)  # 查看 targets 的结构

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    num_class = 21
    val_loss = validate(model, val_loader, device, num_class)

    print(f"Epoch {epoch+1}, Train loss: {train_loss}, Validation loss: {val_loss}")
    # 更新学习率
    lr_scheduler.step()

    # 验证阶段可以在这里添加
    # ...

print("Training complete")

torch.save(model.state_dict(), "faster_rcnn_model.pth")

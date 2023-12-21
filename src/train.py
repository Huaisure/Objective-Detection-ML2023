'''
Discription: train the model
Author: Huaishuo Liu
Maintainer: Huaishuo Liu
Created: 2023-12-20
'''
import torch
import torchvision
from torchvision.transforms import functional as F
from model import create_faster_rcnn_model
from data_preprocessing import get_data_loaders
from torchvision.ops import box_iou


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def validate(model, data_loader, device):
    model.eval()  # 将模型设置为评估模式
    val_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
            print("output:",loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(data_loader)


def train_transforms(img, target):
    # 添加数据增强步骤
    img = F.to_tensor(img)
    return img, target

def val_transforms(img, target):
    img = F.to_tensor(img)
    return img, target

train_loader, val_loader = get_data_loaders('../data', train_transforms, val_transforms, batch_size=4)

num_classes = 21  # 20 类 + 1 背景类
model = create_faster_rcnn_model(num_classes).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    # 显示训练过程
    for images, targets in train_loader:
        print(targets)  # 查看 targets 的结构    

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    val_loss = validate(model, val_loader, device)

    print(f"Epoch {epoch+1}, Train loss: {train_loss}, Validation loss: {val_loss}")
    # 更新学习率
    lr_scheduler.step()

    # 验证阶段可以在这里添加
    # ...

print("Training complete")

torch.save(model.state_dict(), 'faster_rcnn_model.pth')

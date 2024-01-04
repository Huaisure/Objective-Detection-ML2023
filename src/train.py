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
import tqdm
from utils import *
import argparse


def train(load_model, train_transforms, val_transforms, path_to_model, verbose):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader = get_data_loaders(
        "./data", train_transforms, val_transforms, batch_size=4, verbose=verbose
    )

    num_classes = 21  # 20 类 + 1 背景类

    if load_model:
        model = create_faster_rcnn_model(num_classes).to(device)
        # 这里的路径需要根据实际情况修改
        model.load_state_dict(torch.load(path_to_model))
    else:
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
            

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            print(targets)  # 查看 targets 的结构
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-l", action="store_true", help="load model")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    path_to_model = (
        "/home/stu8/workspace/Objective-Detectio-ML2023/faster_rcnn_model.pth"
    )

    args = parser.parse_args()
    train(
        args.load_model, train_transforms, val_transforms, path_to_model, args.verbose
    )


if __name__ == "__main__":
    main()

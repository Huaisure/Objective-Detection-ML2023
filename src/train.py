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
from matplotlib import pyplot as plt
from datetime import datetime


def train(
    load_model=False,
    train_transforms=None,
    val_transforms=None,
    path_to_model=None,
    verbose=False,
    num_epochs=10,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader = get_data_loaders(
        "./data", train_transforms, val_transforms, batch_size=16, verbose=verbose
    )

    num_classes = 21  # 20 类 + 1 背景类

    if load_model:
        model = create_faster_rcnn_model(num_classes).to(device)
        model.load_state_dict(torch.load(path_to_model))
    else:
        model = create_faster_rcnn_model(num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    val_maps = []
    train_losses = []
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
            # print(targets)  # 查看 targets 的结构
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        num_class = 21
        val_map = validate(model, val_loader, device, num_class)
        val_maps.append(val_map)
        train_losses.append(train_loss)

        print(f"Epoch {epoch+1}, Train loss: {train_loss}, Validation mAP: {val_map}")
        # 将结果保存到文件中，result_任务开始时间
        with open(f"result_{time}.txt", "a") as f:
            f.write(
                f"Epoch {epoch+1}, Train loss: {train_loss}, Validation mAP: {val_map}\n"
            )
        f.close()
        # 更新学习率
        lr_scheduler.step()

        # 早期停止, 以train loss为指标
        if epoch > 5:
            if np.mean(train_losses[-2:]) > np.mean(train_losses[-4:-2]):
                print("Early stop")
                num_epochs = epoch + 1
                break

    print("Training complete")
    torch.save(model.state_dict(), f"faster_rcnn_model_{time}.pth")

    # 保存训练结果,绘制图像，横坐标为epoch，纵坐标为validation mAP
    plt.plot(range(1, num_epochs + 1), val_maps)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP")
    plt.savefig(f"result_{time}_val_map.jpg")
    plt.clf()

    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.savefig(f"result_{time}_train_loss.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", "-l", type=str, default=None, help="load_model")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    parser.add_argument("--num_epochs", "-n", type=int, default=10, help="num_epochs")
    # 根据实际情况修改路径
    args = parser.parse_args()
    load = False
    path_to_model = None
    if args.load_model:
        path_to_model = args.load_model
        load = True

    args = parser.parse_args()
    train(
        load_model=load,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        path_to_model=path_to_model,
        verbose=args.verbose,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()

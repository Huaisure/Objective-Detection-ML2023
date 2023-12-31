# 建立文档时的基本信息写在下面
"""
Description: This file contains the data preprocessing for the application.
Author: Huaishuo Liu
Maintainer: Huaishuo Liu
Created: 2023-12-20
"""
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

label_map = {
    "background": 0,  # 通常为背景类别添加一个额外的 ID
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}

reverse_label_map = {v: k for k, v in label_map.items()}  # 反转 label_map


def parse_voc_xml(file):
    """
    parse the xml file
    return: a dictionary containing the bounding boxes and labels
    """
    tree = ET.parse(file)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.iter("object"):
        # difficult means the object is difficult to recognize, so we ignore it
        difficult = int(obj.find("difficult").text)
        if difficult == 1:
            continue
        label = obj.find("name").text
        bnd_box = obj.find("bndbox")
        xmin = int(bnd_box.find("xmin").text)
        ymin = int(bnd_box.find("ymin").text)
        xmax = int(bnd_box.find("xmax").text)
        ymax = int(bnd_box.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return {"boxes": boxes, "labels": labels}


class CustomVOCDataset(Dataset):
    def __init__(self, img_dir, anno_dir, file_ids, transforms=None, input_size=500):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.file_ids = file_ids
        self.transforms = transforms
        self.size = (input_size, input_size)

    def __getitem__(self, idx):
        img_id = self.file_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        anno_path = os.path.join(self.anno_dir, img_id + ".xml")

        # make sure the format of the image is RGB
        img = Image.open(img_path).convert("RGB")
        annotation = parse_voc_xml(anno_path)

        # convert labels to numbers present in the label_map
        labels = [label_map[label] for label in annotation["labels"]]

        target = {}
        target["boxes"] = torch.as_tensor(annotation["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        # print("target:", target)
        # 图像和边界框调整
        # img, target = transform_image_and_boxes(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target, size=self.size)

        return img, target

    def __len__(self):
        return len(self.file_ids)


def split_dataset(base_dir):
    img_dir = os.path.join(base_dir, "JPEGImages")

    all_img_ids = [file.split(".")[0] for file in os.listdir(img_dir)]
    all_img_ids.sort()

    train_ids = [
        img_id for img_id in all_img_ids if "2007_000559" <= img_id <= "2012_001051"
    ]
    val_ids = [img_id for img_id in all_img_ids if img_id not in train_ids]

    return train_ids, val_ids


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    images = torch.stack(images, dim=0)
    return images, targets


def get_data_loaders(
    base_dir, train_transforms, val_transforms, batch_size=4, verbose=True, input_size=500
):
    train_ids, val_ids = split_dataset(base_dir)

    # 查看数据集划分情况
    if verbose:
        print(f"Number of training examples: {len(train_ids)}")
        print(f"Number of validation examples: {len(val_ids)}")

    train_dataset = CustomVOCDataset(
        img_dir=os.path.join(base_dir, "JPEGImages"),
        anno_dir=os.path.join(base_dir, "Annotations"),
        file_ids=train_ids,
        transforms=train_transforms,
        input_size=input_size,
    )

    val_dataset = CustomVOCDataset(
        img_dir=os.path.join(base_dir, "JPEGImages"),
        anno_dir=os.path.join(base_dir, "Annotations"),
        file_ids=val_ids,
        transforms=val_transforms,
        input_size=input_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader

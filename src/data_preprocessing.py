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

def parse_voc_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.iter('object'):
        # difficult means the object is difficult to recognize, so we ignore it
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        label = obj.find('name').text
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return {'boxes': boxes, 'labels': labels}

class CustomVOCDataset(Dataset):
    def __init__(self, img_dir, anno_dir, file_ids, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.file_ids = file_ids
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.file_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        anno_path = os.path.join(self.anno_dir, img_id + ".xml")

        img = Image.open(img_path).convert("RGB")
        annotation = parse_voc_xml(anno_path)

        # 你需要根据你的分类名转换为类别id
        # labels = [your_label_map[label] for label in annotation['labels']]
        labels = annotation['labels']

        target = {}
        target["boxes"] = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        # 添加其他所需字段

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.file_ids)

def split_dataset(base_dir):
    img_dir = os.path.join(base_dir, 'JPEGImages')
    anno_dir = os.path.join(base_dir, 'Annotations')

    all_img_ids = [file.split('.')[0] for file in os.listdir(img_dir)]
    all_img_ids.sort()

    train_ids = [img_id for img_id in all_img_ids if '2007_000559' <= img_id <= '2012_001051']
    val_ids = [img_id for img_id in all_img_ids if img_id not in train_ids]

    return train_ids, val_ids

def get_data_loaders(base_dir, train_transforms, val_transforms, batch_size=4):
    train_ids, val_ids = split_dataset(base_dir)

    train_dataset = CustomVOCDataset(
        img_dir=os.path.join(base_dir, 'JPEGImages'),
        anno_dir=os.path.join(base_dir, 'Annotations'),
        file_ids=train_ids,
        transforms=train_transforms
    )

    val_dataset = CustomVOCDataset(
        img_dir=os.path.join(base_dir, 'JPEGImages'),
        anno_dir=os.path.join(base_dir, 'Annotations'),
        file_ids=val_ids,
        transforms=val_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


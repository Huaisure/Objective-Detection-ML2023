from model import create_faster_rcnn_model
import torch
from utils import *
from data_preprocessing import get_data_loaders

path_to_model = "/home/stu8/workspace/Objective-Detectio-ML2023/faster_rcnn_model.pth"

train_loader, val_loader = get_data_loaders(
    "./data", train_transforms, val_transforms, batch_size=4, verbose=False
)

# 根据训练时的类别数创建模型
num_classes = 21  # 示例：20个类别 + 1个背景类别
model = create_faster_rcnn_model(num_classes)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 加载预训练模型
model.load_state_dict(
    torch.load("/home/stu8/workspace/Objective-Detectio-ML2023/faster_rcnn_model.pth")
)
model.to(device)
model.eval()


val_loss = validate(model, val_loader, device, num_classes)

print("val_map:", val_loss)
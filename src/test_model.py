import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from model import create_faster_rcnn_model
from data_preprocessing import reverse_label_map

# 根据训练时的类别数创建模型
num_classes = 21  # 示例：20个类别 + 1个背景类别
model = create_faster_rcnn_model(num_classes)

# 加载预训练模型
model.load_state_dict(
    torch.load("/home/stu8/workspace/Objective-Detectio-ML2023/faster_rcnn_model.pth")
)
model.eval()


# 准备图片
img = Image.open("./data/JPEGImages/2012_004292.jpg").convert("RGB")
img = F.resize(img, (800, 800))
img_tensor = F.to_tensor(img)


# 预测
with torch.no_grad():
    prediction = model([img_tensor])[0]
    print("########res:", prediction)

# 可视化结果
draw = ImageDraw.Draw(img)
for element in range(len(prediction["boxes"])):
    boxes = prediction["boxes"][element].cpu().numpy()
    score = prediction["scores"][element].cpu().numpy()
    label = prediction["labels"][element].cpu().numpy()
    # label 为numpy数组，需要转换为 Python int
    label = int(label)
    label = reverse_label_map[label]  

    # 仅可视化得分高于某个阈值的预测
    if score > 0.5:
        box = boxes.astype(int)
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red")
        draw.text((box[0], box[1]), f"Label: {label}, Score: {score:.2f}")

# 显示图像
img.save("output.jpg")

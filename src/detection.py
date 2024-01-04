import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from model import create_faster_rcnn_model
from data_preprocessing import reverse_label_map
import argparse


def detection(path_to_image, path_to_model, verbose=False):
    # 根据训练时的类别数创建模型
    num_classes = 21  # 示例：20个类别 + 1个背景类别
    model = create_faster_rcnn_model(num_classes)
    # 加载预训练模型
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    # 准备图片
    img = Image.open(path_to_image).convert("RGB")
    # img = F.resize(img, (800, 800))
    img_tensor = F.to_tensor(img)

    # 预测
    with torch.no_grad():
        prediction = model([img_tensor])[0]
        if verbose:
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

    # 根据输入图像的名称保存图片,保存在当前目录下
    img_name = path_to_image.split("/")[-1].split(".")[0]
    img.save(f"{img_name}_prediction.jpg")
    print(
        "Done! Prediction saved as", f"{img_name}_prediction.jpg in current directory"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_image", "-i", type=str, default=None, help="path_to_image"
    )
    parser.add_argument(
        "--path_to_model",
        "-m",
        type=str,
        default="/home/stu8/workspace/Objective-Detectio-ML2023/faster_rcnn_model.pth",
        help="path_to_model",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    # 根据实际情况修改路径
    args = parser.parse_args()
    if args.path_to_image == None:
        print("please input path_to_image")
    path_to_model = args.path_to_model
    path_to_image = args.path_to_image
    # 测试时使用
    # path_to_image ="./data/JPEGImages/2012_004113.jpg"
    detection(
        path_to_image=path_to_image, path_to_model=path_to_model, verbose=args.verbose
    )

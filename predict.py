import torch
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
from model import TeaPotClassifier

# 定义预处理流程（与验证集相同）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes, device='cuda'):
    model = TeaPotClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device='cuda'):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    # 中心裁剪到最小边
    w, h = image.size
    crop_size = min(w, h)
    image = transforms.functional.center_crop(image, crop_size)
    # 应用验证集的transform
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度
    # 预测
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probabilities, 1)
    return preds.item(), conf.item()

if __name__ == '__main__':
    image_files = [
        'IMG_20240628_141001_1.jpg',
        'IMG_20240628_140750_1.jpg',
        'IMG_20240628_140826_1.jpg'
    ]
    model_path = 'my_model.pth'
    csv_file = 'ocr_results_revised.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取类别映射
    data = pd.read_csv(csv_file)
    class_to_idx = {cls: i for i, cls in enumerate(sorted(data['hu_name'].unique()))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # 加载模型
    model = load_model(model_path, num_classes, device)

    # 对每张图片进行预测
    for img_file in image_files:
        img_path = os.path.join('materials for display', img_file)  # 替换为图片目录
        if not os.path.exists(img_path):
            print(f"Image {img_file} not found!")
            continue
        pred_idx, confidence = predict_image(model, img_path, val_transform, device)
        pred_class = idx_to_class[pred_idx]
        print(f"Image: {img_file} -> Predicted: {pred_class}, Confidence: {confidence:.4f}")

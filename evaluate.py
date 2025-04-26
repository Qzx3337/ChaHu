# metrics.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Precision, Recall
import pandas as pd
import os
from tqdm import tqdm
from dataset import TeaPotDataset
from model import TeaPotClassifier


# 加载模型
def load_model(model_path, num_classes):
    model = TeaPotClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


# 计算指标
def calculate_metrics(model, test_loader):
    device = next(model.parameters()).device
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    # 转换为tensor
    preds_tensor = torch.tensor(all_preds)
    targets_tensor = torch.tensor(all_targets)

    # 计算指标
    f1 = F1Score(task='multiclass', num_classes=num_classes)(preds_tensor, targets_tensor)
    precision = Precision(task='multiclass', num_classes=num_classes)(preds_tensor, targets_tensor)
    recall = Recall(task='multiclass', num_classes=num_classes)(preds_tensor, targets_tensor)

    return {
        'F1': f1.item(),
        'Precision': precision.item(),
        'Recall': recall.item()
    }


# 保存结果
def save_results(metrics, save_path='results.txt'):
    with open(save_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


if __name__ == "__main__":
    # 参数配置
    CSV_PATH = "ocr_results_revised.csv"
    ROOT_DIR = "zisha teapot dataset"
    MODEL_PATH = "my_model.pth"

    # 获取类别数量
    data = pd.read_csv(CSV_PATH)
    num_classes = len(data['hu_name'].unique())
    data = None

    # 初始化组件
    test_dataset = TeaPotDataset(
        root_dir=ROOT_DIR,
        csv_file=CSV_PATH,
        mode='val'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_model(MODEL_PATH, num_classes)
    metrics = calculate_metrics(model, test_loader)
    save_results(metrics)

    print("Metrics saved to results.txt")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")


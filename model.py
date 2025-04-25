# model.py
import torch
import torch.nn as nn
from torchvision import models


class TeaPotClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # EfficientNet，适用高分辨率图像
        self.backbone = models.efficientnet_b3(pretrained=True)

        # 冻结底层参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 替换最后的分类层
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

        # 注意力机制
        # self.attention = nn.Sequential(
        #     nn.Conv2d(1536, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1536, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone.features(x)
        att = self.attention(features)
        features = features * att
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return self.backbone.classifier(features)

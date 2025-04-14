# dataset.py
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TeaPotDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, mode='train'):
        self.data = pd.read_csv(os.path.join(csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = self._get_class_dict()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 数据增强
        if transform is None:
            self.train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.transform = self.train_transform if mode == 'train' else self.val_transform

    def _get_class_dict(self):
        classes = sorted(self.data['hu_name'].unique())
        return {cls: i for i, cls in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        # 自动背景处理，中心裁剪保留主体
        w, h = image.size
        crop_size = min(w, h)
        image = transforms.functional.center_crop(image, crop_size)

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[self.data.iloc[idx, 1]]
        return image, label
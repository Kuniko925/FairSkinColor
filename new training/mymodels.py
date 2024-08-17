import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import Adamax
import torchvision.models as models
from torchvision.models import efficientnet_b3
from torchvision.models.efficientnet import EfficientNet_B3_Weights
from torchvision.models.vgg import VGG16_Weights
from torchvision.models import ResNet50_Weights

class TransDataset(Dataset):
    def __init__(self, dataframe, img_size, ycol, transform=None):
        self.dataframe = dataframe
        self.img_size = img_size
        self.ycol = ycol
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    # idx　前から何番目か
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx][self.ycol]
        masked_img_path = self.dataframe.iloc[idx]["masked filepath"]
        masked_image = Image.open(masked_img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Torchのため
        masked_transform = transforms.Compose([
            transforms.ToTensor(),])
        masked_image = masked_transform(masked_image)

        return image, torch.tensor(label, dtype=torch.float32), masked_image


class EfficientB3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(1536, 1) #classifier[1]で出力されてくるのが1536次元。efficientnet_b3でデフォルトになっている。

    def forward(self, x):
        x = self.base_model(x)
        return x

class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.base_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.base_model.classifier[6] = nn.Linear(4096, 1) #VGG16 の構造から最終層は[6]

    def forward(self, x):
        x = self.base_model(x)
        return x

class ResNet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.base_model.fc.weight.shape[1], 1)
    def forward(self, x):
        x = self.base_model(x)
        return x
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特徴量抽出
        self.feature = nn.Sequential(
            nn.Conv2d(1, 7, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2)
            )
        
        # 分類
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

        def forward(selfd, x):
            x = self.feature(x)
            x = self.classifier(x)
            return x

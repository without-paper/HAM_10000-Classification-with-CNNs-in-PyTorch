import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.4):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        # Add Dropout between the fully connected layers
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet18(x)
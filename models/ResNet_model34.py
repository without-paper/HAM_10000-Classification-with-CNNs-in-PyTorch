import torch.nn as nn
from torchvision import models


class ResNet34(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.4):
        super(ResNet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)

        # Add Dropout between the fully connected layers
        in_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet34(x)
import torch.nn as nn
from torchvision import models


class ResNet101(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.4):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)

        # Add Dropout between the fully connected layers
        in_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet101(x)
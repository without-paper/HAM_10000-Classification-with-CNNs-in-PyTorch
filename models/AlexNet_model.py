import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.4):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)

        # Adjust the fully connected layer to fit the output
        in_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.alexnet(x)
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
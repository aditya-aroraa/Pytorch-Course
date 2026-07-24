import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Load pretrained ResNet-50 with V2 weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Replace the final fully connected layer with Dropout + Linear
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
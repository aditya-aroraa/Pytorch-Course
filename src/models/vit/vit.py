import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.3
        )

    def forward(self, x):
        return self.model(x)
import torch.nn as nn
import timm


class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.3
        )

    def forward(self, x):
        return self.model(x)
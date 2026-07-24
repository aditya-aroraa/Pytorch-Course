"""
Standalone script to generate Grad-CAM / Grad-CAM++ visualizations
for all models that already have a best_model.pth checkpoint.
"""

import os
import torch

from src.config import DEVICE, IMAGE_SIZE, NUM_CLASSES
from src.dataset import get_dataloaders
from src.gradcam import generate_gradcam_images

# Model imports
from src.models.resnet50.resnet50 import ResNet50Classifier
from src.models.densenet121.densenet121 import DenseNet121Classifier
from src.models.efficientnet_b0.efficientnet_b0 import EfficientNetB0Classifier
from src.models.vgg16.vgg16 import VGG16Classifier
from src.models.vit.vit import ViTClassifier
from src.models.swin.swin import SwinTransformerClassifier


MODEL_REGISTRY = {
    "resnet50": ResNet50Classifier,
    "densenet121": DenseNet121Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "vgg16": VGG16Classifier,
    "vit": ViTClassifier,
    "swin": SwinTransformerClassifier,
}


def main():
    _, _, test_loader = get_dataloaders()

    for model_name, ModelClass in MODEL_REGISTRY.items():
        output_dir = os.path.join("outputs", model_name)
        checkpoint = os.path.join(output_dir, "best_model.pth")

        if not os.path.exists(checkpoint):
            print(f"Skipping {model_name} (no checkpoint found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Generating Grad-CAM for: {model_name.upper()}")
        print(f"{'=' * 60}")

        model = ModelClass(num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model.eval()

        generate_gradcam_images(
            model=model,
            dataset=test_loader.dataset,
            device=DEVICE,
            output_dir=output_dir,
            image_size=IMAGE_SIZE,
            samples_per_class=3
        )

    print("\nDone!")


if __name__ == "__main__":
    main()

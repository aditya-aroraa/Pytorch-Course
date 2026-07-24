import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluate import evaluate_model

# Choose ONE model
from src.models.resnet50.resnet50 import ResNet50Classifier
# from src.models.densenet121.densenet121 import DenseNet121Classifier
# from src.models.efficientnet_b0.efficientnet_b0 import EfficientNetB0Classifier
# from src.models.vit.vit import ViTClassifier
# from src.models.swin.swin import SwinTransformerClassifier
# from src.models.cvt.cvt import CvT21Classifier
# from src.models.vgg16.vgg16 import VGG16Classifier

from src.dataset import get_dataloaders
from src.trainer import train_model
from src.config import DEVICE, LEARNING_RATE, NUM_EPOCHS


# ==========================================
# Load Dataset
# ==========================================

train_loader, val_loader, test_loader = get_dataloaders()


# ==========================================
# Initialize Model
# ==========================================

model = ResNet50Classifier().to(DEVICE)
# model = DenseNet121Classifier().to(DEVICE)
# model = EfficientNetB0Classifier().to(DEVICE)
# model = ViTClassifier().to(DEVICE)
# model = SwinTransformerClassifier().to(DEVICE)
# model = CvT21Classifier().to(DEVICE)
# model = VGG16Classifier().to(DEVICE)


# ==========================================
# Loss Function
# ==========================================

import numpy as np

# Dynamically calculate class weights from training dataset label distribution
train_labels = train_loader.dataset.labels
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)
num_classes = len(class_counts)
class_weights = total_samples / (num_classes * class_counts)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

print(f"Dataset summary - Class counts: {class_counts}")
print(f"Calculated class-weights for loss function: {class_weights.cpu().numpy()}")

# CrossEntropyLoss with dynamic class weights and 0.1 label smoothing
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)


# ==========================================
# Optimizer and Scheduler
# ==========================================

# Upgrade to AdamW with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# Introduce CosineAnnealingLR scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6
)


# ==========================================
# Train Model
# ==========================================

# Dynamically determine the active model name to configure the output directory
model_class_name = model.__class__.__name__.lower()
model_name = model_class_name.replace("classifier", "").replace("transformer", "")
output_dir = f"outputs/{model_name}"
best_model_path = f"{output_dir}/best_model.pth"

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    epochs=NUM_EPOCHS,
    output_dir=output_dir,
    scheduler=scheduler
)


# ==========================================
# Load Best Model
# ==========================================

print("\nLoading Best Model...\n")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)


# ==========================================
# Evaluate Model
# ==========================================

evaluate_model(
    model,
    test_loader,
    DEVICE,
    output_dir
)
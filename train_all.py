import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEVICE, IMAGE_SIZE
from src.preprocess import create_clean_dataset
from src.dataset import get_dataloaders
from src.trainer import train_model
from src.evaluate import evaluate_model
from src.gradcam import generate_gradcam_images

# Model Imports
from src.models.resnet50.resnet50 import ResNet50Classifier
from src.models.densenet121.densenet121 import DenseNet121Classifier
from src.models.efficientnet_b0.efficientnet_b0 import EfficientNetB0Classifier
from src.models.vgg16.vgg16 import VGG16Classifier
from src.models.vit.vit import ViTClassifier
from src.models.swin.swin import SwinTransformerClassifier


def get_model_config(model_key, num_classes=3):
    """
    Returns model and config for 2-phase training.
    Phase 1: Freeze backbone, train only head
    Phase 2: Unfreeze all, differential LR
    """
    if model_key == "resnet50":
        model = ResNet50Classifier(num_classes=num_classes)
        head_param_names = ["fc"]
        lr_head = 1e-3
        lr_backbone_phase2 = 3e-5
        lr_head_phase2 = 3e-4

    elif model_key == "densenet121":
        model = DenseNet121Classifier(num_classes=num_classes)
        head_param_names = ["classifier"]
        lr_head = 1e-3
        lr_backbone_phase2 = 3e-5
        lr_head_phase2 = 3e-4

    elif model_key == "efficientnet_b0":
        model = EfficientNetB0Classifier(num_classes=num_classes)
        head_param_names = ["classifier"]
        lr_head = 1e-3
        lr_backbone_phase2 = 3e-5
        lr_head_phase2 = 3e-4

    elif model_key == "vgg16":
        model = VGG16Classifier(num_classes=num_classes)
        head_param_names = ["classifier.6"]
        lr_head = 5e-4
        lr_backbone_phase2 = 1e-5
        lr_head_phase2 = 2e-4

    elif model_key == "vit":
        model = ViTClassifier(num_classes=num_classes)
        head_param_names = ["model.head"]
        lr_head = 1e-3
        lr_backbone_phase2 = 1e-5
        lr_head_phase2 = 2e-4

    elif model_key == "swin":
        model = SwinTransformerClassifier(num_classes=num_classes)
        head_param_names = ["model.head"]
        lr_head = 1e-3
        lr_backbone_phase2 = 1.5e-5
        lr_head_phase2 = 2e-4

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    return model, head_param_names, lr_head, lr_backbone_phase2, lr_head_phase2


def is_head_param(name, head_param_names):
    """Check if a parameter name belongs to the head."""
    return any(h in name for h in head_param_names)


def freeze_backbone(model, head_param_names):
    """Freeze all parameters except the head."""
    for name, param in model.named_parameters():
        if is_head_param(name, head_param_names):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def main():
    print("=" * 60)
    print("STEP 1: PREPARING OUTPUT DIRECTORY")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)
    print("Output directory ready (preserving existing results).")

    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING RAW ULTRASOUND DATASET")
    print("=" * 60)
    create_clean_dataset()

    print("\n" + "=" * 60)
    print("STEP 3: LOADING DATALOADERS")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_dataloaders()

    # Calculate class weights
    train_labels = train_loader.dataset.labels
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    print(f"Train samples: {len(train_labels)} | Class distribution: {class_counts}")
    print(f"Calculated class weights: {class_weights.cpu().numpy().round(3)}")

    models_to_train = [
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "swin",
        "vgg16",
        "vit"
    ]

    phase1_epochs = 10
    phase2_epochs = 30

    for idx, m_key in enumerate(models_to_train, 1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(models_to_train)}] TRAINING MODEL: {m_key.upper()}")
        print("=" * 60)

        output_dir = os.path.join("outputs", m_key)
        os.makedirs(output_dir, exist_ok=True)

        # Skip models that already have completed results
        metrics_path = os.path.join(output_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            print(f"  => SKIPPING {m_key} (results already exist)")
            continue

        model, head_param_names, lr_head, lr_bb_p2, lr_head_p2 = get_model_config(m_key, num_classes)
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # ==================================================
        # PHASE 1: Freeze backbone, train head only
        # ==================================================
        print(f"\n--- Phase 1: Training HEAD only ({phase1_epochs} epochs) ---")

        freeze_backbone(model, head_param_names)

        head_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {sum(p.numel() for p in head_params):,} "
              f"(out of {sum(p.numel() for p in model.parameters()):,} total)")

        optimizer_p1 = optim.AdamW(head_params, lr=lr_head, weight_decay=1e-2)
        scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p1, T_max=phase1_epochs, eta_min=1e-5
        )

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_p1,
            device=DEVICE,
            epochs=phase1_epochs,
            output_dir=output_dir,
            scheduler=scheduler_p1,
            patience=999,  # No early stopping in phase 1
            max_grad_norm=1.0
        )

        # ==================================================
        # PHASE 2: Unfreeze all, fine-tune with differential LR
        # ==================================================
        print(f"\n--- Phase 2: Fine-tuning ALL layers ({phase2_epochs} epochs) ---")

        # Load best model from phase 1 as starting point
        best_p1_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_p1_path):
            model.load_state_dict(torch.load(best_p1_path, map_location=DEVICE))

        unfreeze_all(model)

        backbone_params = [p for n, p in model.named_parameters()
                           if not is_head_param(n, head_param_names)]
        head_params_p2 = [p for n, p in model.named_parameters()
                          if is_head_param(n, head_param_names)]

        print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}")
        print(f"  Head params: {sum(p.numel() for p in head_params_p2):,}")

        optimizer_p2 = optim.AdamW([
            {"params": backbone_params, "lr": lr_bb_p2},
            {"params": head_params_p2, "lr": lr_head_p2}
        ], weight_decay=1e-2)

        scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p2, T_max=phase2_epochs, eta_min=1e-6
        )

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_p2,
            device=DEVICE,
            epochs=phase2_epochs,
            output_dir=output_dir,
            scheduler=scheduler_p2,
            patience=12,
            max_grad_norm=1.0
        )

        # ==================================================
        # EVALUATE with TTA
        # ==================================================
        best_model_path = os.path.join(output_dir, "best_model.pth")
        print(f"\nEvaluating Best Checkpoint for {m_key} on Test Set (with TTA)...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=DEVICE,
            output_dir=output_dir,
            use_tta=True,
            image_size=IMAGE_SIZE
        )

        # ==================================================
        # GENERATE GRAD-CAM VISUALIZATIONS
        # ==================================================
        print(f"\nGenerating Grad-CAM / Grad-CAM++ for {m_key}...")
        generate_gradcam_images(
            model=model,
            dataset=test_loader.dataset,
            device=DEVICE,
            output_dir=output_dir,
            image_size=IMAGE_SIZE,
            samples_per_class=3
        )

    print("\n" + "=" * 60)
    print("STEP 4: GENERATING COMPARISON PLOTS")
    print("=" * 60)
    import subprocess
    subprocess.run([".venv/bin/python", "plot_comparison.py"], check=True)

    print("\nALL MODELS TRAINED AND RESULTS SAVED IN 'outputs' DIRECTORY SUCCESSFULLY!")


if __name__ == "__main__":
    main()

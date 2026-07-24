import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from src.config import CLASSES


# =====================================================
# TEST-TIME AUGMENTATION (TTA)
# =====================================================

def get_tta_transforms(image_size=224):
    """
    Returns a list of TTA transforms.
    Each produces a different view of the same image.
    The original (no augmentation) is always included.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tta_list = [
        # Original
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        # Both flips
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            normalize
        ]),
    ]

    return tta_list


@torch.no_grad()
def evaluate_model(model,
                   test_loader,
                   device,
                   output_dir,
                   use_tta=True,
                   image_size=224):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []  # Store softmax probabilities for AUC-ROC

    if use_tta:
        tta_transforms = get_tta_transforms(image_size)
        num_tta = len(tta_transforms)

        # We need raw PIL images, so iterate over the dataset directly
        dataset = test_loader.dataset

        for idx in range(len(dataset)):
            image_path = dataset.image_paths[idx]
            label = dataset.labels[idx]

            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")

            # Accumulate softmax predictions across all TTA views
            avg_probs = None

            for tta_t in tta_transforms:
                img_tensor = tta_t(pil_image).unsqueeze(0).to(device)

                if str(device).startswith("cuda"):
                    with torch.amp.autocast("cuda"):
                        output = model(img_tensor)
                else:
                    output = model(img_tensor)

                probs = F.softmax(output, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = avg_probs + probs

            avg_probs = avg_probs / num_tta
            _, predicted = torch.max(avg_probs, 1)

            y_true.append(label)
            y_pred.append(predicted.cpu().item())
            y_prob.append(avg_probs.cpu().numpy().flatten())

    else:
        for images, labels in test_loader:
            images = images.to(device)

            if str(device).startswith("cuda"):
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        digits=4
    )

    print("\n==============================")
    print("TEST RESULTS" + (" (with TTA)" if use_tta else ""))
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix\n")
    print(cm)

    print("\nClassification Report\n")
    print(report)

    # ---------------------------------
    # Save Classification Report
    # ---------------------------------

    with open(
        os.path.join(output_dir, "classification_report.txt"),
        "w"
    ) as f:

        f.write("==============================\n")
        f.write("TEST RESULTS" + (" (with TTA)\n" if use_tta else "\n"))
        f.write("==============================\n\n")

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")

        f.write("Confusion Matrix\n\n")
        f.write(str(cm))
        f.write("\n\nClassification Report\n\n")
        f.write(report)

    # ---------------------------------
    # Save Metrics CSV
    # ---------------------------------

    with open(
        os.path.join(output_dir, "metrics.csv"),
        "w"
    ) as f:

        f.write("Accuracy,Precision,Recall,F1\n")
        f.write(
            f"{accuracy:.4f},"
            f"{precision:.4f},"
            f"{recall:.4f},"
            f"{f1:.4f}\n"
        )

    # ---------------------------------
    # Save Confusion Matrix Figure
    # ---------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title("Confusion Matrix" + (" (TTA)" if use_tta else ""))

    plt.savefig(
        os.path.join(output_dir, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------
    # AUC-ROC Curve (One-vs-Rest)
    # ---------------------------------

    y_prob_np = np.array(y_prob)
    y_true_np = np.array(y_true)
    num_classes = y_prob_np.shape[1]

    # Binarize true labels for one-vs-rest
    y_true_bin = label_binarize(y_true_np, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    # Per-class ROC curves
    all_auc = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_np[:, i])
        roc_auc = auc(fpr, tpr)
        class_name = CLASSES[i] if i < len(CLASSES) else f"Class {i}"
        all_auc[class_name] = roc_auc
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{class_name} (AUC = {roc_auc:.4f})"
        )

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_true_bin[:, i], y_prob_np[:, i])[0] for i in range(num_classes)]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob_np[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    all_auc["macro-average"] = macro_auc

    ax.plot(
        all_fpr, mean_tpr,
        color="#333333",
        linewidth=2.5,
        linestyle="--",
        label=f"Macro-average (AUC = {macro_auc:.4f})"
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "AUC-ROC Curve (One-vs-Rest)" + (" — with TTA" if use_tta else ""),
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_dir, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Save AUC scores to CSV
    with open(os.path.join(output_dir, "auc_scores.csv"), "w") as f:
        f.write("Class,AUC\n")
        for cls_name, auc_val in all_auc.items():
            f.write(f"{cls_name},{auc_val:.4f}\n")

    print("\nAUC-ROC Scores:")
    for cls_name, auc_val in all_auc.items():
        print(f"  {cls_name:>15}: {auc_val:.4f}")
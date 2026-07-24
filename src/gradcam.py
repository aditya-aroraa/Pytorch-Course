"""
Grad-CAM and Grad-CAM++ for Explainable AI.

Generates class-discriminative heatmaps that highlight which regions of the
ultrasound image the model focuses on for its prediction.

References:
    - Grad-CAM:   Selvaraju et al., 2017  (arXiv:1610.02391)
    - Grad-CAM++: Chattopadhyay et al., 2018  (arXiv:1710.11063)
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import CLASSES, IMAGE_SIZE


# =====================================================================
# Target Layer Auto-Detection
# =====================================================================

def get_target_layer(model):
    """
    Automatically detects the correct last convolutional / feature-map
    layer for the supported model architectures.

    Returns:
        target_layer (nn.Module): The layer to hook for Grad-CAM.
        reshape_transform (callable or None): Optional function to reshape
            transformer feature maps into spatial (B, C, H, W) format.
    """
    inner = model.model  # all wrappers store torchvision/timm model as .model

    class_name = inner.__class__.__name__.lower()

    # ------------------------------------------------------------------
    # CNN models — straightforward last conv block
    # ------------------------------------------------------------------
    if "resnet" in class_name:
        return inner.layer4[-1], None

    if "densenet" in class_name:
        return inner.features.denseblock4, None

    if "efficientnet" in class_name:
        return inner.features[-1], None

    if "vgg" in class_name:
        return inner.features[-1], None

    # ------------------------------------------------------------------
    # Vision Transformers — need reshape from (B, N, C) → (B, C, H, W)
    # ------------------------------------------------------------------
    if "visiontransformer" in class_name:
        # timm ViT: blocks[-1].norm1 gives activations before attention
        target = inner.blocks[-1].norm1

        def vit_reshape(tensor):
            # tensor shape: (B, N, C) where N = 1 (cls) + H*W patches
            # Remove CLS token and reshape to spatial grid
            B, N, C = tensor.shape
            # ViT patch16 on 224x224 → 14x14 = 196 patches + 1 cls = 197
            h = w = int((N - 1) ** 0.5)
            return tensor[:, 1:, :].permute(0, 2, 1).reshape(B, C, h, w)

        return target, vit_reshape

    if "swin" in class_name:
        # timm Swin: layers[-1].blocks[-1].norm1
        target = inner.layers[-1].blocks[-1].norm1

        def swin_reshape(tensor):
            # Swin norm1 output shape: (B, H, W, C)
            if tensor.dim() == 4:
                return tensor.permute(0, 3, 1, 2)  # (B, C, H, W)
            # Fallback for (B, N, C)
            B, N, C = tensor.shape
            h = w = int(N ** 0.5)
            return tensor.permute(0, 2, 1).reshape(B, C, h, w)

        return target, swin_reshape

    raise ValueError(
        f"Unsupported model architecture: {inner.__class__.__name__}. "
        f"Please add target layer detection for this model."
    )


# =====================================================================
# Grad-CAM
# =====================================================================

class GradCAM:
    """
    Standard Grad-CAM.

    Computes the class-discriminative heatmap by global-average-pooling
    the gradients flowing into the target convolutional layer.
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image tensor.
            target_class: Class index to visualize.  If None, uses the
                          predicted class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations  # (B, C, h, w) or (B, N, C)
        gradients = self.gradients      # same shape

        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# =====================================================================
# Grad-CAM++
# =====================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++.

    Uses higher-order gradients to compute per-pixel weights, providing
    better localization especially when multiple object instances exist.
    """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

        self.activations = None
        self.gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """
        Compute Grad-CAM++ heatmap.

        Args:
            input_tensor: (1, C, H, W) input image tensor.
            target_class: Class index to visualize.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if self.reshape_transform is not None:
            activations = self.reshape_transform(activations)
            gradients = self.reshape_transform(gradients)

        # Grad-CAM++ alpha computation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Sum of activations over spatial dims for each channel
        spatial_sum = activations.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Denominator: 2 * grad^2 + spatial_sum * grad^3
        denom = 2.0 * grad_2 + spatial_sum * grad_3
        denom = torch.where(
            denom != 0,
            denom,
            torch.ones_like(denom)
        )

        alpha = grad_2 / denom  # (B, C, h, w)

        # Weights: sum over spatial of (alpha * relu(grad))
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# =====================================================================
# Visualization Utilities
# =====================================================================

def _apply_heatmap(image_np, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay a Grad-CAM heatmap onto an image.

    Args:
        image_np: (H, W, 3) RGB image in [0, 255], uint8.
        cam: (h, w) heatmap in [0, 1].
        alpha: Overlay transparency.
        colormap: OpenCV colormap for the heatmap.

    Returns:
        overlay: (H, W, 3) RGB image, uint8.
    """
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap) * alpha + np.float32(image_np) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def _get_preprocess_transform(image_size):
    """Standard ImageNet preprocessing for inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =====================================================================
# Main Entry Point
# =====================================================================

def generate_gradcam_images(model,
                            dataset,
                            device,
                            output_dir,
                            image_size=IMAGE_SIZE,
                            samples_per_class=3,
                            seed=42):
    """
    Generates Grad-CAM and Grad-CAM++ visualization figures for a sample
    of test images and saves them.

    For each sampled image, creates a side-by-side figure:
        Original  |  Grad-CAM  |  Grad-CAM++

    Args:
        model: Trained classifier (in eval mode).
        dataset: A BUSIDataset instance (needs .image_paths and .labels).
        device: 'cuda' or 'cpu'.
        output_dir: Base model output directory (e.g. 'outputs/resnet50').
        image_size: Input image size for the model.
        samples_per_class: How many images to sample from each class.
        seed: Random seed for reproducible sampling.
    """
    gradcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    model.eval()

    # Detect target layer
    target_layer, reshape_transform = get_target_layer(model)

    # Instantiate both methods
    cam_method = GradCAM(model, target_layer, reshape_transform)
    cam_pp_method = GradCAMPlusPlus(model, target_layer, reshape_transform)

    # Preprocessing for model input
    preprocess = _get_preprocess_transform(image_size)

    # ---------------------------------------------------------------
    # Sample images: pick `samples_per_class` per class
    # ---------------------------------------------------------------
    rng = random.Random(seed)

    indices_by_class = {}
    for idx in range(len(dataset)):
        label = dataset.labels[idx]
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(idx)

    sampled_indices = []
    for label in sorted(indices_by_class.keys()):
        pool = indices_by_class[label]
        n = min(samples_per_class, len(pool))
        sampled_indices.extend(rng.sample(pool, n))

    print(f"\n  Generating Grad-CAM visualizations for {len(sampled_indices)} images...")

    # ---------------------------------------------------------------
    # Generate heatmaps
    # ---------------------------------------------------------------
    for count, idx in enumerate(sampled_indices, 1):
        image_path = dataset.image_paths[idx]
        true_label = dataset.labels[idx]
        true_class = CLASSES[true_label]

        # Load original image (for display)
        pil_image = Image.open(image_path).convert("RGB")
        display_image = pil_image.resize((image_size, image_size))
        image_np = np.array(display_image)

        # Preprocess for model
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Forward pass to get predicted class
        with torch.no_grad():
            pred = model(input_tensor).argmax(dim=1).item()
        pred_class = CLASSES[pred]

        # Compute heatmaps (target = predicted class)
        cam_heatmap = cam_method(input_tensor, target_class=pred)
        cam_pp_heatmap = cam_pp_method(input_tensor, target_class=pred)

        # Create overlays
        overlay_cam = _apply_heatmap(image_np, cam_heatmap)
        overlay_cam_pp = _apply_heatmap(image_np, cam_pp_heatmap)

        # ---------------------------------------------------------------
        # Plot side-by-side: Original | Grad-CAM | Grad-CAM++
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title(f"Original\nTrue: {true_class}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(overlay_cam)
        axes[1].set_title(f"Grad-CAM\nPred: {pred_class}", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(overlay_cam_pp)
        axes[2].set_title(f"Grad-CAM++\nPred: {pred_class}", fontsize=12)
        axes[2].axis("off")

        fig.suptitle(
            f"{os.path.basename(image_path)}  —  True: {true_class} | Pred: {pred_class}",
            fontsize=13,
            fontweight="bold",
            y=1.02
        )

        plt.tight_layout()

        save_name = f"{true_class}_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        save_path = os.path.join(gradcam_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"    [{count}/{len(sampled_indices)}] Saved: {save_name}")

    # Cleanup hooks
    cam_method.remove_hooks()
    cam_pp_method.remove_hooks()

    print(f"  Grad-CAM outputs saved to: {gradcam_dir}")

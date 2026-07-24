from pathlib import Path

import cv2
import numpy as np

from src.config import ROOT_DIR, CLASSES

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RAW_DATASET_DIR = ROOT_DIR / "Dataset_BUSI_with_GT"
OUTPUT_DIR = ROOT_DIR / "Dataset_BUSI_Clean"

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
IMAGE_SIZE = (256, 256)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def get_image_files(folder: Path):
    """
    Returns only original BUSI images.
    Ignores all segmentation mask images.
    """
    images = []
    if not folder.exists():
        return images

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in VALID_EXTENSIONS:
            continue

        if "_mask" in file.stem:
            continue

        images.append(file)

    return sorted(images)


def preprocess_image(image_path: Path):
    """
    Applies high-quality preprocessing to a BUSI ultrasound image.

    Pipeline:
      1. Convert BGR → RGB
      2. Resize to target size with high-quality interpolation
      3. Bilateral filter to reduce speckle noise while preserving edges
      4. CLAHE on the L-channel in LAB space for contrast enhancement
      5. Gentle unsharp masking to sharpen lesion boundaries
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize cleanly
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # -------------------------------------------------
    # Bilateral Filter — reduce ultrasound speckle noise
    # while preserving edge structures
    # -------------------------------------------------
    image = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    # -------------------------------------------------
    # Enhanced CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Stronger clipLimit for better contrast in ultrasound
    # -------------------------------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # -------------------------------------------------
    # Gentle Unsharp Masking — sharpen lesion boundaries
    # -------------------------------------------------
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    image = cv2.addWeighted(image, 1.3, gaussian, -0.3, 0)

    return image


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------


def create_clean_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    print("=" * 60)
    print("CREATING CLEAN DATASET FROM RAW SOURCE (Dataset_BUSI_with_GT)")
    print("=" * 60)

    for cls in CLASSES:
        source_folder = RAW_DATASET_DIR / cls
        destination_folder = OUTPUT_DIR / cls
        destination_folder.mkdir(parents=True, exist_ok=True)

        copied = 0

        for image_path in get_image_files(source_folder):
            image = preprocess_image(image_path)

            if image is None:
                continue

            save_path = destination_folder / image_path.name

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), image)

            copied += 1
            total += 1

        print(f"{cls:<12}: {copied} images")

    print("-" * 60)
    print(f"Total Images : {total}")
    print(f"Saved To     : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    create_clean_dataset()
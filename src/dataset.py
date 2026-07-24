from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from sklearn.model_selection import train_test_split

from src.config import *


# =====================================================
# DATA AUGMENTATION
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# CUSTOM DATASET
# =====================================================

class BUSIDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")

        label = self.labels[index]

        if self.transform:

            image = self.transform(image)

        return image, label


# =====================================================
# LOAD ALL IMAGES
# =====================================================

def load_dataset():

    image_paths = []

    labels = []

    label_map = {
        "benign": 0,
        "malignant": 1,
        "normal": 2
    }

    for cls in CLASSES:

        folder = DATASET_DIR / cls

        for file in sorted(folder.iterdir()):

            if "_mask" in file.stem:
                continue

            if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            image_paths.append(file)

            labels.append(label_map[cls])

    return image_paths, labels


# =====================================================
# CREATE DATALOADERS
# =====================================================

def get_dataloaders():

    image_paths, labels = load_dataset()

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    train_dataset = BUSIDataset(
        train_paths,
        train_labels,
        train_transform
    )

    val_dataset = BUSIDataset(
        val_paths,
        val_labels,
        test_transform
    )

    test_dataset = BUSIDataset(
        test_paths,
        test_labels,
        test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
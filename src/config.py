from pathlib import Path

# ======================================================
# PROJECT PATHS
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "Dataset_BUSI_Clean"

OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
METRIC_DIR = OUTPUT_DIR / "metrics"

# ======================================================
# DATASET
# ======================================================

CLASSES = [
    "benign",
    "malignant",
    "normal"
]

NUM_CLASSES = len(CLASSES)

# ======================================================
# IMAGE SETTINGS
# ======================================================

IMAGE_SIZE = 224

# ======================================================
# TRAINING SETTINGS
# ======================================================

BATCH_SIZE = 16

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

RANDOM_SEED = 42

DEVICE = "cuda"
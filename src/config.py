"""System-wide configuration: hyperparameters, image sizes, and paths.

Environment variables are set at the very top, before any TensorFlow import,
to silence the oneDNN/abseil INFO noise on startup.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

from logger import setup_logging

setup_logging()  # initialise console + logs/pipeline.log handlers

# ---------------------------------------------------------------------------
# Project root & directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = RAW_DIR / "train"
VAL_DIR = RAW_DIR / "val"
TEST_DIR = RAW_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
DIAGNOSTIC_CSV = OUTPUTS_DIR / "diagnostic_results.csv"
CASES_DIR = OUTPUTS_DIR / "cases"

SRC_DIR = PROJECT_ROOT / "src"

# Ensure output directories always exist.
for _d in (DATA_DIR, RAW_DIR, PREPROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, CASES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image parameters
# ---------------------------------------------------------------------------
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
GRAYSCALE = True          # chest X-rays are single channel
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 6
VALIDATION_SPLIT = 0.15

# Transfer-learning backbone toggle: "custom" or "mobilenetv2"
MODEL_BACKBONE = "mobilenetv2"

MODEL_PATH = MODELS_DIR / "medical_image_analyzer.keras"

# ---------------------------------------------------------------------------
# Preprocessing / augmentation
# ---------------------------------------------------------------------------
AUGMENTATION = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest",
}

# ---------------------------------------------------------------------------
# Simulated clinical metadata pools (used for dashboard / CSV records)
# ---------------------------------------------------------------------------
SCANNER_MANUFACTURERS = ["Siemens", "GE", "Philips"]
GENDERS = ["Male", "Female"]
AGE_RANGE = (1, 90)

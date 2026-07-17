"""Image preprocessing pipeline.

import _env  # must be first: silence TF startup logs


Provides three concerns:

1. **Offline preprocessing** - convert raw chest X-rays to grayscale, resize,
   enhance contrast with CLAHE, normalise, and persist the results to
   ``data/preprocessed/`` so the Streamlit PACS dashboard can render the
   original vs. preprocessed comparison side-by-side.
2. **NumPy/OpenCV helpers** - ``preprocess_for_inference`` is reused by the
   prediction pipeline so uploaded images receive exactly the same treatment
   as the training data.
3. **TensorFlow augmentation** - ``build_augmentation`` returns a Keras layer
   used for on-the-fly augmentation during training (the raw dataset is also
   preprocessed on-the-fly inside ``train.py``).

All functions assume single-channel (grayscale) chest X-ray input.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import (
    AUGMENTATION,
    CLASS_NAMES,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    IMG_SIZE,
    PREPROCESSED_DIR,
    RAW_DIR,
)
from logger import get_logger

logger = get_logger(__name__)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    ``image`` is a uint8 grayscale array of shape (H, W) or (H, W, 1).
    """
    if image.ndim == 3:
        image = image.squeeze(-1)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    return clahe.apply(image)


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """Full preprocessing for a single BGR/RGB or grayscale uint8 image.

    Steps: grayscale -> resize -> CLAHE -> normalise to [0, 1] -> (H, W, 1).
    Returns a float32 array ready for model input.
    """
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)

    image = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]))
    image = apply_clahe(image)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=-1)


def load_preprocessed(image_path: Path) -> np.ndarray:
    """Load an already-preprocessed (CLAHE'd) grayscale PNG and return a
    batched float32 tensor normalised to [0, 1].

    Use this for images from ``data/processed/`` that already had CLAHE
    applied during offline preprocessing.  Do **not** re-apply CLAHE.
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    raw = cv2.resize(raw, (IMG_SIZE[1], IMG_SIZE[0]))
    arr = raw.astype(np.float32) / 255.0
    return np.expand_dims(np.expand_dims(arr, axis=-1), axis=0)


def preprocess_for_inference(image_path: Path) -> np.ndarray:
    """Load a **raw** (not previously processed) image from disk, apply the
    full preprocessing chain (grayscale → resize → CLAHE → normalise), and
    return a batched float32 tensor.

    Use this for user-uploaded images that have not been through the offline
    preprocessing pipeline.
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    proc = preprocess_array(raw)
    return np.expand_dims(proc, axis=0)


def build_augmentation() -> tf.keras.Sequential:
    """Keras augmentation stack mirroring ``config.AUGMENTATION``."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                AUGMENTATION["rotation_range"] / 360.0
            ),
            tf.keras.layers.RandomZoom(
                (-AUGMENTATION["zoom_range"], AUGMENTATION["zoom_range"])
            ),
            tf.keras.layers.RandomTranslation(
                AUGMENTATION["height_shift_range"],
                AUGMENTATION["width_shift_range"],
            ),
            tf.keras.layers.RandomFlip("horizontal"),
        ]
    )


def save_preprocessed_dataset(raw_root: Path = RAW_DIR) -> dict:
    """Preprocess every raw image and persist to ``data/preprocessed/``.

    Preserves the ``<split>/<class>/`` structure. Returns a summary dict of
    processed counts. Images are saved as 8-bit grayscale PNGs.
    """
    summary = {}
    for split in ("train", "val", "test"):
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for class_name in CLASS_NAMES:
            src_dir = split_dir / class_name
            dst_dir = PREPROCESSED_DIR / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            count = 0
            for img_path in sorted(src_dir.iterdir()):
                if img_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }:
                    continue
                raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    continue
                proc = preprocess_array(raw)
                out = (proc.squeeze(-1) * 255.0).astype(np.uint8)
                cv2.imwrite(str(dst_dir / (img_path.stem + ".png")), out)
                count += 1
            summary[f"{split}/{class_name}"] = count
            logger.info("Preprocessed %s/%s: %d images", split, class_name, count)
    return summary


def main() -> None:
    logger.info("Starting offline preprocessing of data/raw -> %s", PREPROCESSED_DIR)
    summary = save_preprocessed_dataset()
    total = sum(summary.values())
    lines = ["=" * 50, "PREPROCESSING COMPLETE".center(50), "=" * 50]
    for key, value in summary.items():
        lines.append(f"  {key:<20}{value:>8} images")
    lines += ["-" * 50, f"  {'TOTAL':<20}{total:>8} images",
              f"Saved to: {PREPROCESSED_DIR}"]
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()

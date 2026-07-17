"""Real-time inference pipeline with Grad-CAM explainability.

import _env  # must be first: silence TF startup logs


Exposes ``predict`` which, given an image path, runs the trained classifier,
returns the predicted class and confidence, and produces a Grad-CAM heatmap
overlay that highlights the lung regions (opacity locations) that most
influenced the decision - an XAI aid for radiologists.

Also persists every prediction to ``outputs/diagnostic_results.csv`` together
with simulated clinical metadata (age, gender, scanner) and processing
latency, feeding the historical case table in the dashboards.
"""

import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import (
    AGE_RANGE,
    CASES_DIR,
    CLASS_NAMES,
    DIAGNOSTIC_CSV,
    GENDERS,
    IMG_SIZE,
    MODEL_PATH,
    PLOTS_DIR,
    SCANNER_MANUFACTURERS,
)
from logger import get_logger

logger = get_logger(__name__)

CSV_FIELDS = [
    "timestamp",
    "patient_id",
    "age",
    "gender",
    "scanner",
    "filename",
    "prediction",
    "confidence",
    "latency_ms",
]

_rng = random.Random(42)


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Trained model not found at {MODEL_PATH}. Run `train.py` first."
        )
    return tf.keras.models.load_model(str(MODEL_PATH))


def _build_gradcam_model(model: tf.keras.Model):
    """Build a Grad-CAM sub-model targeting the final conv feature map.

    Prefers a top-level ``Conv2D`` directly connected to ``model.input``
    (e.g. the ``gradcam_conv`` layer added before the head). Falls back to a
    nested Functional backbone if no top-level conv exists.
    """
    top_convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if top_convs:
        conv_layer = top_convs[-1]
        return tf.keras.models.Model(model.input, [conv_layer.output, model.output])

    # Fallback: nested backbone (contains conv layers).
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            isinstance(l, tf.keras.layers.Conv2D)
            for l in layer._flatten_layers(include_self=False, recursive=True)
        ):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    convs = [
        l
        for l in backbone._flatten_layers(include_self=False, recursive=True)
        if isinstance(l, tf.keras.layers.Conv2D)
    ]
    return tf.keras.models.Model(backbone.input, [convs[-1].output, model.output])


def _compute_heatmap(model, image_batch, grad_model, class_idx=None):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image_batch)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(pooled * conv_out[0], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def _make_overlay(original_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a jet-coloured heatmap over the original RGB image."""
    h, w = original_rgb.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    blended = (0.6 * original_rgb.astype(np.float32) + 0.4 * jet.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _load_threshold() -> float:
    """Operational decision threshold (Youden's J) produced by evaluate.py."""
    p = PLOTS_DIR.parent / "optimal_threshold.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text())["optimal_threshold"])
        except Exception:
            return 0.5
    return 0.5


def predict(
    image_path: Path,
    model: tf.keras.Model | None = None,
    preprocessed: bool = False,
) -> dict:
    """Run inference + Grad-CAM on a single image.

    Args:
        image_path: Path to the image file.
        model: Optional pre-loaded Keras model.
        preprocessed: If ``True``, the image is already CLAHE-preprocessed
            (from ``data/processed/``) and should be loaded without
            re-applying CLAHE.  If ``False`` (default), the full
            preprocessing chain (resize + CLAHE + normalise) is applied,
            suitable for raw user uploads.

    Returns a dict with keys: class_name, confidence, overlay (RGB uint8),
    preprocessed (H,W,1 float32), original (RGB uint8).
    """
    if model is None:
        model = load_model()

    t0 = time.perf_counter()

    original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    if preprocessed:
        from preprocess import load_preprocessed
        image_batch = load_preprocessed(image_path)
    else:
        from preprocess import preprocess_for_inference
        image_batch = preprocess_for_inference(image_path)

    probs = model.predict(image_batch, verbose=0)[0]
    pneu_idx = CLASS_NAMES.index("PNEUMONIA")
    pneu_prob = float(probs[pneu_idx])
    threshold = _load_threshold()
    class_idx = pneu_idx if pneu_prob >= threshold else (1 - pneu_idx)
    confidence = pneu_prob if class_idx == pneu_idx else 1.0 - pneu_prob

    grad_model = _build_gradcam_model(model)
    heatmap = _compute_heatmap(model, image_batch, grad_model, class_idx)
    overlay = _make_overlay(original_rgb, heatmap)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "class_name": CLASS_NAMES[class_idx],
        "confidence": confidence,
        "overlay": overlay,
        "preprocessed": image_batch[0],
        "original": original_rgb,
        "latency_ms": latency_ms,
    }
    return result


def _simulate_metadata(filename: str) -> dict:
    return {
        "patient_id": f"PX-{_rng.randint(10000, 99999)}",
        "age": _rng.randint(*AGE_RANGE),
        "gender": _rng.choice(GENDERS),
        "scanner": _rng.choice(SCANNER_MANUFACTURERS),
        "filename": Path(filename).name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def save_case_artifacts(result: dict, meta: dict, image_path: Path) -> dict:
    """Persist the three PACS views (original/preprocessed/Grad-CAM) for a case.

    Returns a dict of filenames written (relative to ``CASES_DIR``) so the
    dashboard can re-load them for a selected patient.
    """
    case_dir = CASES_DIR / meta["patient_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    original = result["original"].astype(np.uint8)
    preprocessed = (result["preprocessed"].squeeze(-1) * 255.0).astype(np.uint8)
    overlay = result["overlay"].astype(np.uint8)

    files = {
        "original": f"{meta['patient_id']}_original.png",
        "preprocessed": f"{meta['patient_id']}_preprocessed.png",
        "gradcam": f"{meta['patient_id']}_gradcam.png",
    }
    cv2.imwrite(str(case_dir / files["original"]), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(case_dir / files["preprocessed"]), preprocessed)
    cv2.imwrite(str(case_dir / files["gradcam"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return files


def record_prediction(result: dict, image_path: Path) -> dict:
    """Append a prediction plus simulated metadata to the CSV database.

    Also persists the three PACS views under ``outputs/cases/<patient_id>/``.
    Returns the metadata record that was generated.
    """
    meta = _simulate_metadata(str(image_path))
    row = {
        "timestamp": meta["timestamp"],
        "patient_id": meta["patient_id"],
        "age": meta["age"],
        "gender": meta["gender"],
        "scanner": meta["scanner"],
        "filename": meta["filename"],
        "prediction": result["class_name"],
        "confidence": f"{result['confidence']:.4f}",
        "latency_ms": f"{result['latency_ms']:.1f}",
    }
    write_header = not DIAGNOSTIC_CSV.exists()
    with DIAGNOSTIC_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    save_case_artifacts(result, meta, image_path)
    return meta


def predict_and_record(image_path: Path, preprocessed: bool = False) -> dict:
    """Convenience: predict then persist the diagnostic record."""
    result = predict(image_path, preprocessed=preprocessed)
    record_prediction(result, image_path)
    return result


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    preprocessed = "--preprocessed" in args
    args = [a for a in args if a != "--preprocessed"]

    if len(args) < 1:
        print("Usage: python predict.py [--preprocessed] <image_path>")
        raise SystemExit(1)
    target = Path(args[0])
    if not target.exists():
        logger.error("File not found: %s", target)
        raise SystemExit(1)
    out = predict_and_record(target, preprocessed=preprocessed)
    logger.info("Prediction : %s", out["class_name"])
    logger.info("Confidence : %.4f", out["confidence"])
    logger.info("Latency    : %.1f ms", out["latency_ms"])
    logger.info("Recorded to: %s", DIAGNOSTIC_CSV)

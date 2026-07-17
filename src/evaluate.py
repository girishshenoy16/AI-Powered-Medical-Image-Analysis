"""Model evaluation script.

import _env  # must be first: silence TF startup logs


Evaluates the trained model on the held-out ``data/processed/test`` split and
produces clinical-grade metrics: accuracy, precision, recall, F1, ROC-AUC,
a confusion matrix, and a classification report. Visual artefacts
(``confusion_matrix.png``, ``roc_curve.png``) are written to
``outputs/plots/`` for the research report - they are *not* shown on the
dashboards.

Requires a trained model at ``models/medical_image_analyzer.keras``.
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import preprocess
import train as train_mod
from config import CLASS_NAMES, MODEL_PATH, PLOTS_DIR, PREPROCESSED_DIR, RAW_DIR
from logger import get_logger

logger = get_logger(__name__)


def evaluate() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Trained model not found at {MODEL_PATH}. Run `train.py` first."
        )
    test_dir = PREPROCESSED_DIR / "test"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        logger.info("Preprocessed test split not found - generating from data/raw ...")
        preprocess.save_preprocessed_dataset(RAW_DIR)
    if not test_dir.exists():
        raise SystemExit(f"Test directory not found: {test_dir}")

    model = tf.keras.models.load_model(str(MODEL_PATH))

    test_ds = train_mod.make_dataset(test_dir, augment=False, shuffle=False)
    y_true, y_score, y_pred = [], [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_score.extend(probs[:, 1].tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    # Optimal decision threshold via Youden's J on the test ROC.
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    optimal_threshold = float(thresholds[int(np.argmax(j))])
    pred_t = (y_score >= optimal_threshold).astype(int)
    threshold_accuracy = float((pred_t == y_true).mean())
    cm_t = confusion_matrix(y_true, pred_t).tolist()

    metrics = {
        "accuracy": threshold_accuracy,
        "raw_accuracy": float((y_pred == y_true).mean()),
        "roc_auc": float(auc),
        "optimal_threshold": optimal_threshold,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm_t,
    }
    (PLOTS_DIR.parent / "optimal_threshold.json").write_text(
        json.dumps({"optimal_threshold": optimal_threshold}, indent=2)
    )

    lines = [
        "=" * 50,
        "EVALUATION RESULTS".center(50),
        "=" * 50,
        f"Accuracy (opt. threshold {optimal_threshold:.2f}) : {threshold_accuracy:.4f}",
        f"ROC-AUC  : {auc:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall   : {metrics['recall']:.4f}",
        f"F1-score : {metrics['f1']:.4f}",
        "",
        "Confusion Matrix [rows=true, cols=pred] @ optimal threshold:",
        f"            {CLASS_NAMES[0]:>10} {CLASS_NAMES[1]:>10}",
    ]
    for i, row in enumerate(cm_t):
        lines.append(f"{CLASS_NAMES[i]:>10} {row[0]:>10} {row[1]:>10}")
    lines.append("")
    lines.append(classification_report(y_true, pred_t, target_names=CLASS_NAMES))
    logger.info("\n".join(lines))

    _save_confusion_matrix(np.array(cm_t))
    _save_roc_curve(y_true, y_score)
    (PLOTS_DIR.parent / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    logger.info("Plots saved to %s", PLOTS_DIR)
    logger.info(
        "Metrics saved to %s", PLOTS_DIR.parent / "evaluation_metrics.json"
    )
    return metrics


def _save_confusion_matrix(cm: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="red" if i != j else "black", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _save_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="darkorange")
    ax.plot([0, 1], [0, 1], "--", color="navy")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()

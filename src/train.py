"""Model training script.

Loads the **preprocessed** scans from ``data/processed`` (CLAHE contrast
enhancement is applied once, offline, by ``preprocess.py`` and persisted as
PNGs). The training pipeline only normalises to [0, 1] and applies augmentation
on-the-fly and normalises. CLAHE is applied once, offline, so it is never
recomputed per epoch. This is dramatically faster than per-epoch on-the-fly
CLAHE on CPU.

Outputs:
* best weights -> ``models/medical_image_analyzer.keras`` (checkpointing),
* accuracy/loss curves -> ``outputs/plots/training_history.png``,
* ``outputs/training_history.json`` log for downstream reporting.

Early stopping guards against over-fitting. The dataset is validated up-front
via ``verify_dataset`` so the run fails fast with a clear message. If the
preprocessed split is missing, the script regenerates it automatically.
"""

import json

import tensorflow as tf

import preprocess
from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IMG_SIZE,
    MODEL_PATH,
    PLOTS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
    VAL_DIR,
)
from logger import get_logger
from model import get_model
from verify_dataset import verify_dataset

logger = get_logger(__name__)

HISTORY_JSON = PLOTS_DIR.parent / "training_history.json"
THRESHOLD_JSON = PLOTS_DIR.parent / "optimal_threshold.json"


def compute_class_weights() -> dict:
    """Balance the 74/26 train split so the minority (NORMAL) class is not
    drowned out. Returns ``{class_index: weight}`` for ``model.fit``."""
    counts = {}
    for i, class_name in enumerate(CLASS_NAMES):
        d = PREPROCESSED_DIR / "train" / class_name
        counts[i] = max(1, sum(1 for _ in d.glob("*.png")))
    total = sum(counts.values())
    return {i: total / (2 * counts[i]) for i, _ in counts.items()}


class EpochLogger(tf.keras.callbacks.Callback):
    """Mirror Keras epoch progress into the application logger."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {logs.get('loss', 0):.4f}"
            f" - accuracy: {logs.get('accuracy', 0):.4f}"
        )
        if "val_loss" in logs:
            msg += (
                f" - val_loss: {logs['val_loss']:.4f}"
                f" - val_accuracy: {logs['val_accuracy']:.4f}"
            )
        logger.info(msg)


def _count_samples(ds) -> int:
    """Estimate the number of examples in a batched tf.data.Dataset."""
    total = 0
    for batch in ds:
        total += tf.shape(batch[0])[0].numpy()
    return int(total)


def make_dataset(directory, augment=False, shuffle=False, cache=False):
    """Build a normalised, optionally augmented tf.data.Dataset.

    ``directory`` is expected to contain ``NORMAL/`` and ``PNEUMONIA/``
    sub-folders of already-CLAHE-enhanced grayscale PNGs.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    # Preprocessed PNGs already had CLAHE applied; just scale to [0, 1].
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if cache:
        ds = ds.cache()
    if augment:
        aug = preprocess.build_augmentation()
        ds = ds.map(
            lambda x, y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def _ensure_preprocessed() -> None:
    """Regenerate ``data/processed`` if the train split is missing."""
    if (PREPROCESSED_DIR / "train").exists() and any(
        (PREPROCESSED_DIR / "train").iterdir()
    ):
        return
    logger.info("Preprocessed split not found - generating from data/raw ...")
    preprocess.save_preprocessed_dataset(RAW_DIR)


def train() -> dict:
    report = verify_dataset()
    if not report["ready"]:
        raise SystemExit(
            "Dataset not ready. Place images under "
            "data/raw/{train,val,test}/{NORMAL,PNEUMONIA}."
        )

    _ensure_preprocessed()

    logger.info("Building model...")
    model = get_model()
    model.summary(print_fn=logger.info)

    logger.info("Loading datasets...")
    train_ds = make_dataset(
        PREPROCESSED_DIR / "train", augment=True, shuffle=True
    )
    val_dir = PREPROCESSED_DIR / "val"
    val_ds = (
        make_dataset(val_dir, augment=False, shuffle=False)
        if val_dir.exists() and any(val_dir.iterdir())
        else None
    )

    class_weights = compute_class_weights()
    logger.info("Class weights (counter imbalance): %s", class_weights)

    # With only 16 validation images, val metrics are too noisy to drive
    # checkpointing/early stopping, so fall back to training accuracy.
    monitor = "val_accuracy" if val_ds and _count_samples(val_ds) >= 50 else "accuracy"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor=monitor,
        save_best_only=True,
        verbose=0,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    logger.info("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        shuffle=False,  # train_ds is a tf.data.Dataset already shuffled
        verbose=0,
        class_weight=class_weights,
        callbacks=[EpochLogger(), checkpoint, early_stop],
    ).history

    serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
    HISTORY_JSON.write_text(json.dumps(serialisable, indent=2))

    _save_history_plot(history)

    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("History log saved to %s", HISTORY_JSON)
    return serialisable


def _save_history_plot(history: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="Train Accuracy")
    if val_acc:
        axes[0].plot(val_acc, label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss, label="Train Loss")
    if val_loss:
        axes[1].plot(val_loss, label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_history.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train()

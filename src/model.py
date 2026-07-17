"""Deep learning model architectures.

import _env  # must be first: silence TF startup logs


Two backbones are provided:

* ``build_custom_cnn`` - a lightweight, from-scratch convolutional network
  suited to the modest 5.8k-image chest X-ray dataset.
* ``build_mobilenetv2`` - transfer learning using a frozen
  MobileNetV2 ImageNet backbone with a trainable classification head.

``get_model`` dispatches based on ``config.MODEL_BACKBONE``.
"""

import tensorflow as tf

from config import IMG_SIZE, LEARNING_RATE, MODEL_BACKBONE, NUM_CLASSES


def _compile(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn() -> tf.keras.Model:
    """A compact 4-block CNN with global average pooling."""
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input")
    x = inputs
    filters = [32, 64, 128, 256]
    for f in filters:
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    return _compile(tf.keras.Model(inputs, outputs, name="custom_cnn"))


def build_mobilenetv2() -> tf.keras.Model:
    """MobileNetV2 transfer-learning head on grayscale input.

    The backbone is adapted to a single channel by replicating the input
    across 3 channels (MobileNetV2 expects RGB). The backbone weights are
    frozen; only the classification head is trained.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input")
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    # Explicit top-level conv layer retained as the Grad-CAM target so the
    # heatmap is directly connected to model.input (the backbone is nested).
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu",
                               name="gradcam_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    return _compile(tf.keras.Model(inputs, outputs, name="mobilenetv2"))


def get_model() -> tf.keras.Model:
    backbone = MODEL_BACKBONE.lower()
    if backbone == "custom":
        return build_custom_cnn()
    if backbone == "mobilenetv2":
        return build_mobilenetv2()
    raise ValueError(f"Unknown MODEL_BACKBONE: {MODEL_BACKBONE}")


if __name__ == "__main__":
    model = get_model()
    model.summary()

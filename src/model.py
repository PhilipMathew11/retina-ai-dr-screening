"""Transfer-learning model factory."""

from __future__ import annotations


def build_model(image_size: int = 224, weights: str | None = "imagenet"):
    """Build a binary EfficientNetB0 model with an optional pretrained backbone."""
    import tensorflow as tf

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="fundus_image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dr_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="aptos_efficientnetb0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

"""Grad-CAM heatmap generation for the EfficientNetB0 classifier."""

from __future__ import annotations

import cv2
import numpy as np


def make_gradcam(
    model, image_batch: np.ndarray, class_index: int | None = None
) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for the requested binary class."""
    import tensorflow as tf

    backbone = model.get_layer("efficientnetb0")
    augmented_input = model.get_layer("training_augmentation")(
        image_batch, training=False
    )
    conv_output = backbone(augmented_input, training=False)
    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        pooled = model.get_layer("global_average_pool")(conv_output)
        dropped = model.get_layer("dropout")(pooled, training=False)
        prediction = model.get_layer("dr_probability")(dropped)
        if class_index is None:
            class_index = int(float(prediction[0, 0].numpy()) >= 0.5)
        if class_index not in (0, 1):
            raise ValueError("Grad-CAM class index must be 0 (Normal) or 1 (DR).")
        score = prediction[:, 0] if class_index == 1 else 1 - prediction[:, 0]
    gradients = tape.gradient(score, conv_output)
    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients for the selected layer.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_output, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) == 0:
        return np.zeros((image_batch.shape[1], image_batch.shape[2]), dtype=np.float32)
    heatmap = heatmap / max_value
    return cv2.resize(
        heatmap.numpy(),
        (image_batch.shape[2], image_batch.shape[1]),
        interpolation=cv2.INTER_LINEAR,
    )


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a colorized heatmap over an RGB image."""
    heat = np.uint8(np.clip(heatmap, 0, 1) * 255)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb.astype(np.uint8), 1 - alpha, colored, alpha, 0)

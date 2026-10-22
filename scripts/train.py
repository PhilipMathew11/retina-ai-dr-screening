from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.model import build_model
from src.preprocessing import prepare_model_input


def dataset_from_frame(frame: pd.DataFrame, batch_size: int, training: bool):
    paths = frame["path"].to_numpy(dtype=str)
    labels = frame["label"].to_numpy(dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        def decode(path_value):
            image = cv2.imread(path_value.decode("utf-8"), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read training image: {path_value!r}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return prepare_model_input(image)

        image = tf.numpy_function(decode, [path], tf.float32)
        image.set_shape((224, 224, 3))
        return image, label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(frame), 2048), seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the APTOS EfficientNetB0 binary classifier.")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/dr_screening.keras"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fine-tune", action="store_true")
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    train = splits[splits["split"] == "train"]
    validation = splits[splits["split"] == "validation"]
    if train.empty or validation.empty or set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split file must contain non-empty train, validation, and test partitions.")
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=train["label"].to_numpy())
    class_weights = dict(enumerate(weights.tolist()))

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.model, monitor="val_loss", save_best_only=True
        ),
    ]
    args.model.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        dataset_from_frame(train, args.batch_size, training=True),
        validation_data=dataset_from_frame(validation, args.batch_size, training=False),
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    if args.fine_tune:
        backbone = model.get_layer("efficientnetb0")
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        fine_tune_history = model.fit(
            dataset_from_frame(train, args.batch_size, training=True),
            validation_data=dataset_from_frame(validation, args.batch_size, training=False),
            epochs=max(1, args.epochs // 2),
            class_weight=class_weights,
            callbacks=callbacks,
        )
        for key, values in fine_tune_history.history.items():
            history.history.setdefault(key, []).extend(values)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv("models/training_history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, title in (
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "loss", "Loss"),
    ):
        axis.plot(history_frame[key], label="Training")
        axis.plot(history_frame[f"val_{key}"], label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig("models/training_curves.png", dpi=150)
    plt.close(figure)
    print(f"Best validation-loss checkpoint saved to {args.model}")


if __name__ == "__main__":
    main()

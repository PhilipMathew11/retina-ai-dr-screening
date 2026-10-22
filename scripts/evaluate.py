from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.train import dataset_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model once on the held-out test split.")
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/dr_screening.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--preliminary",
        action="store_true",
        help="Label this evaluation as preliminary for an unfinished training run.",
    )
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    test = splits[splits["split"] == "test"]
    if test.empty:
        raise ValueError("No held-out test examples found.")
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(dataset_from_frame(test, 16, training=False)).ravel()
    y_true = test["label"].to_numpy(dtype=int)
    y_pred = (probabilities >= 0.5).astype(int)
    metrics = {
        "evaluation_scope": (
            "preliminary_checkpoint" if args.preliminary else "final_held_out_test"
        ),
        "training_run_status": "interrupted_before_completion" if args.preliminary else "completed",
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": 0.5,
        "test_samples": int(len(test)),
        "positive_class": "Diabetic Retinopathy (APTOS grade > 0)",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"id_code": test["id_code"], "label": y_true, "probability_dr": probabilities, "prediction": y_pred}
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "DR"],
        yticklabels=["Normal", "DR"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Held-out test confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

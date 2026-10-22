"""Dataset label loading and reproducible stratified split creation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
def build_splits(
    labels_csv: Path,
    image_dir: Path,
    output_csv: Path,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Map APTOS grades to binary labels and persist a stratified 70/15/15 split."""
    labels = _read_labels(labels_csv)
    labels["path"] = labels["id_code"].map(
        lambda image_id: _resolve_image(image_dir, str(image_id))
    )
    labels["label"] = (labels["diagnosis"] > 0).astype("int32")
    train_val, test = train_test_split(
        labels, test_size=0.15, random_state=seed, stratify=labels["label"]
    )
    train, validation = train_test_split(
        train_val,
        test_size=(0.15 / 0.85),
        random_state=seed,
        stratify=train_val["label"],
    )
    return _save_splits(train, validation, test, output_csv)


def build_predefined_splits(
    train_csv: Path,
    validation_csv: Path,
    test_csv: Path,
    train_image_dir: Path,
    validation_image_dir: Path,
    test_image_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Use the dataset's existing labeled splits without repartitioning them."""
    partitions = []
    for split_name, csv_path, image_dir in (
        ("train", train_csv, train_image_dir),
        ("validation", validation_csv, validation_image_dir),
        ("test", test_csv, test_image_dir),
    ):
        labels = _read_labels(csv_path)
        labels["path"] = labels["id_code"].map(
            lambda image_id: _resolve_image(image_dir, str(image_id))
        )
        labels["label"] = (labels["diagnosis"] > 0).astype("int32")
        partitions.append(labels.assign(split=split_name))

    result = pd.concat(partitions, ignore_index=True)
    if result["id_code"].duplicated().any():
        raise ValueError("Image IDs must not appear in more than one dataset split.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result[["id_code", "diagnosis", "label", "split", "path"]].to_csv(
        output_csv, index=False
    )
    return result


def _read_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    required = {"id_code", "diagnosis"}
    if not required.issubset(labels.columns):
        raise ValueError("Labels CSV must contain 'id_code' and 'diagnosis' columns.")
    if (
        labels.empty
        or labels["id_code"].isna().any()
        or labels["id_code"].duplicated().any()
        or not pd.to_numeric(labels["diagnosis"], errors="coerce")
        .between(0, 4)
        .all()
    ):
        raise ValueError("Labels must contain non-empty APTOS grades from 0 through 4.")
    labels["diagnosis"] = labels["diagnosis"].astype("int32")
    return labels


def _save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")
    result = pd.concat((train, validation, test), ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result[["id_code", "diagnosis", "label", "split", "path"]].to_csv(
        output_csv, index=False
    )
    return result


def _resolve_image(image_dir: Path, image_id: str) -> str:
    for suffix in ("", ".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"No image found for dataset ID '{image_id}' in {image_dir}.")

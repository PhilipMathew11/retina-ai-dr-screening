from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import build_predefined_splits, build_splits


def _find_image_dir(data_dir: Path, folder_name: str) -> Path:
    for candidate in (data_dir / folder_name / folder_name, data_dir / folder_name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{folder_name}' or its nested image directory under {data_dir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare binary train/validation/test metadata from APTOS labels."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/splits.csv"))
    args = parser.parse_args()
    train_csv = args.data_dir / "train_1.csv"
    validation_csv = args.data_dir / "valid.csv"
    test_csv = args.data_dir / "test.csv"
    predefined_csvs = (train_csv, validation_csv, test_csv)
    if any(path.exists() for path in predefined_csvs) and not all(
        path.is_file() for path in predefined_csvs
    ):
        missing = [path.name for path in predefined_csvs if not path.is_file()]
        raise FileNotFoundError(
            "The provided split layout is incomplete; missing CSV files: "
            + ", ".join(missing)
        )
    if all(path.is_file() for path in predefined_csvs):
        splits = build_predefined_splits(
            train_csv,
            validation_csv,
            test_csv,
            _find_image_dir(args.data_dir, "train_images"),
            _find_image_dir(args.data_dir, "val_images"),
            _find_image_dir(args.data_dir, "test_images"),
            args.output,
        )
    else:
        labels = args.labels or args.data_dir / "aptos" / "train.csv"
        images = args.images or args.data_dir / "aptos" / "train_images"
        splits = build_splits(labels, images, args.output)
    print("Saved split metadata:", args.output)
    print(pd.crosstab(splits["split"], splits["label"], margins=True))


if __name__ == "__main__":
    main()

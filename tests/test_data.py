import pandas as pd

from src.data import build_predefined_splits, build_splits


def test_build_splits_are_disjoint_and_stratified(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    labels = pd.DataFrame(
        {
            "id_code": [f"image_{index}" for index in range(20)],
            "diagnosis": [0 if index % 2 == 0 else 1 for index in range(20)],
        }
    )
    labels_path = tmp_path / "train.csv"
    labels.to_csv(labels_path, index=False)
    for image_id in labels["id_code"]:
        (image_dir / f"{image_id}.jpg").touch()

    result = build_splits(labels_path, image_dir, tmp_path / "splits.csv")

    assert len(result) == 20
    assert set(result["split"]) == {"train", "validation", "test"}
    assert result.groupby("split")["label"].nunique().eq(2).all()
    assert result["id_code"].is_unique
    assert (tmp_path / "splits.csv").is_file()


def test_predefined_splits_are_preserved_and_checked_for_overlap(tmp_path):
    partitions = {}
    for split, ids in (
        ("train", ("train_a", "train_b")),
        ("validation", ("valid_a", "valid_b")),
        ("test", ("test_a", "test_b")),
    ):
        image_dir = tmp_path / split
        image_dir.mkdir()
        rows = []
        for index, image_id in enumerate(ids):
            (image_dir / f"{image_id}.png").touch()
            rows.append({"id_code": image_id, "diagnosis": index})
        labels_path = tmp_path / f"{split}.csv"
        pd.DataFrame(rows).to_csv(labels_path, index=False)
        partitions[split] = (labels_path, image_dir)

    result = build_predefined_splits(
        partitions["train"][0],
        partitions["validation"][0],
        partitions["test"][0],
        partitions["train"][1],
        partitions["validation"][1],
        partitions["test"][1],
        tmp_path / "splits.csv",
    )

    assert result.groupby("split").size().to_dict() == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert result.groupby("split")["id_code"].apply(set).to_dict() == {
        "train": {"train_a", "train_b"},
        "validation": {"valid_a", "valid_b"},
        "test": {"test_a", "test_b"},
    }


def test_predefined_splits_reject_reused_image_ids(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "shared.png").touch()
    csv_paths = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.csv"
        pd.DataFrame([{"id_code": "shared", "diagnosis": 0}]).to_csv(path, index=False)
        csv_paths.append(path)

    import pytest

    with pytest.raises(ValueError, match="more than one dataset split"):
        build_predefined_splits(
            *csv_paths,
            image_dir,
            image_dir,
            image_dir,
            tmp_path / "splits.csv",
        )

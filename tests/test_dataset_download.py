from pathlib import Path

from scripts.download_kaggle_dataset import normalize_dataset


def test_normalize_dataset_creates_cat_and_dog_folders(tmp_path: Path) -> None:
    dataset_root = tmp_path / "downloaded"
    (dataset_root / "train" / "cats").mkdir(parents=True)
    (dataset_root / "train" / "dogs").mkdir(parents=True)

    (dataset_root / "train" / "cats" / "cat_1.jpg").write_bytes(b"cat")
    (dataset_root / "train" / "cats" / "cat_2.jpg").write_bytes(b"cat")
    (dataset_root / "train" / "dogs" / "dog_1.jpg").write_bytes(b"dog")

    target_root = tmp_path / "data" / "raw"
    normalize_dataset(dataset_root, target_root)

    assert (target_root / "cat").exists()
    assert (target_root / "dog").exists()
    assert len(list((target_root / "cat").iterdir())) == 2
    assert len(list((target_root / "dog").iterdir())) == 1

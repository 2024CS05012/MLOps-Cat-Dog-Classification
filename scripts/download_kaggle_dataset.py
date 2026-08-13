import argparse
import os
import shutil
from pathlib import Path


def find_image_directories(dataset_root: Path) -> dict[str, list[Path]]:
    matches: dict[str, list[Path]] = {"cat": [], "dog": []}
    for root, dirnames, _ in os.walk(dataset_root):
        current = Path(root)
        lower_name = current.name.lower()
        if lower_name in {"cat", "cats", "dog", "dogs", "petimages"}:
            if lower_name in {"cat", "cats"}:
                matches["cat"].append(current)
            if lower_name in {"dog", "dogs"}:
                matches["dog"].append(current)
    return matches


def normalize_dataset(dataset_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    image_dirs = find_image_directories(dataset_root)

    for class_name, dirs in image_dirs.items():
        destination = target_root / class_name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

        for source_dir in dirs:
            for image_path in sorted(source_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    shutil.copy2(image_path, destination / image_path.name)

    if not any((target_root / name).exists() for name in ["cat", "dog"]):
        raise FileNotFoundError(
            f"Could not find cat/dog image folders inside dataset root: {dataset_root}."
        )


def download_dataset(dataset_name: str, output_dir: Path | None = None) -> Path:
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - covered by installation step
        raise RuntimeError("kagglehub is required. Install it with: pip install kagglehub") from exc

    download_root = kagglehub.dataset_download(dataset_name)
    dataset_path = Path(download_root)
    target_root = output_dir or Path("data/raw")
    normalize_dataset(dataset_path, target_root)
    return target_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize the Kaggle cats-vs-dogs dataset.")
    parser.add_argument(
        "--dataset-name",
        default="bhavikjikadara/dog-and-cat-classification-dataset",
        help="Kaggle dataset slug to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination folder for the normalized cat/dog dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.dataset_name, args.output_dir)
    print(f"Dataset downloaded and normalized under: {args.output_dir.resolve()}")

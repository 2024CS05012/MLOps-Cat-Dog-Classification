import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from src.config import CLASS_NAMES, IMAGE_SIZE


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB and fix orientation from EXIF metadata."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def preprocess_image_file(input_path: Path, output_path: Path, size: tuple[int, int] = IMAGE_SIZE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = resize_image(load_rgb_image(input_path), size)
    image.save(output_path, format="JPEG", quality=95)


def split_files(files: list[Path], train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[Path]]:
    shuffled = files[:]
    random.Random(42).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def preprocess_dataset(raw_dir: Path, processed_dir: Path) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / f"{class_name}s"
        if not class_dir.exists():
            class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder for {class_name}: {raw_dir}")

        image_files = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for split_name, split_paths in split_files(image_files).items():
            for index, input_path in enumerate(split_paths):
                output_path = processed_dir / split_name / class_name / f"{class_name}_{index:05d}.jpg"
                preprocess_image_file(input_path, output_path)

from pathlib import Path

from torchvision import datasets, transforms

from src.config import IMAGE_SIZE


def build_transforms(split: str) -> transforms.Compose:
    common = [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                *common,
            ]
        )
    return transforms.Compose(common)


def image_folder_dataset(processed_dir: Path, split: str) -> datasets.ImageFolder:
    return datasets.ImageFolder(processed_dir / split, transform=build_transforms(split))

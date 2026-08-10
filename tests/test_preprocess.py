from pathlib import Path

from PIL import Image

from src.data.preprocess import preprocess_image_file, split_files


def test_preprocess_image_file_resizes_and_converts_rgb(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    target = tmp_path / "output.jpg"
    Image.new("L", (32, 16), color=128).save(source)

    preprocess_image_file(source, target, size=(224, 224))

    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (224, 224)


def test_split_files_is_deterministic() -> None:
    files = [Path(f"image_{index}.jpg") for index in range(10)]

    first = split_files(files)
    second = split_files(files)

    assert first == second
    assert len(first["train"]) == 8
    assert len(first["val"]) == 1
    assert len(first["test"]) == 1

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", type=Path, default=Path("data/sample/cat.jpg"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    if args.image.exists():
        with args.image.open("rb") as image_file:
            files = {"file": (args.image.name, image_file, "image/jpeg")}
            prediction = requests.post(
                f"{args.base_url}/predict",
                files=files,
                timeout=20,
            )
    else:
        buffer = BytesIO()
        Image.new("RGB", (224, 224), color=(120, 80, 40)).save(buffer, format="JPEG")
        buffer.seek(0)
        prediction = requests.post(
            f"{args.base_url}/predict",
            files={"file": ("synthetic-smoke.jpg", buffer, "image/jpeg")},
            timeout=20,
        )
    prediction.raise_for_status()
    print(prediction.json())

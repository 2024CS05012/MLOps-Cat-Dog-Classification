import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import requests

from src.config import CLASS_NAMES, PREDICTIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "simulated_requests.json")
    return parser.parse_args()


def iter_request_images(samples_dir: Path):
    image_paths = sorted(samples_dir.glob("*.jpg"))
    if image_paths:
        for image_path in image_paths:
            true_label = next((label for label in CLASS_NAMES if label in image_path.stem.lower()), "unknown")
            with image_path.open("rb") as image_file:
                yield image_path.name, true_label, image_file.read()
        return

    synthetic_specs = [
        ("simulated-cat.jpg", "cat", (130, 95, 70)),
        ("simulated-dog.jpg", "dog", (90, 115, 145)),
    ]
    for filename, true_label, color in synthetic_specs:
        buffer = BytesIO()
        Image.new("RGB", (224, 224), color=color).save(buffer, format="JPEG")
        yield filename, true_label, buffer.getvalue()


if __name__ == "__main__":
    args = parse_args()
    results = []
    for filename, true_label, image_bytes in iter_request_images(args.samples_dir):
        response = requests.post(
            f"{args.base_url}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=20,
        )
        response.raise_for_status()
        prediction = response.json()
        results.append({"image": filename, "true_label": true_label, **prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} simulated post-deployment predictions to {args.output}")

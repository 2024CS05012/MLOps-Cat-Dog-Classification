import argparse
import json
from pathlib import Path

import requests

from src.config import CLASS_NAMES, PREDICTIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "simulated_requests.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = []
    for image_path in sorted(args.samples_dir.glob("*.jpg")):
        true_label = next((label for label in CLASS_NAMES if label in image_path.stem.lower()), "unknown")
        with image_path.open("rb") as image_file:
            response = requests.post(
                f"{args.base_url}/predict",
                files={"file": (image_path.name, image_file, "image/jpeg")},
                timeout=20,
            )
        response.raise_for_status()
        prediction = response.json()
        results.append({"image": str(image_path), "true_label": true_label, **prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} simulated post-deployment predictions to {args.output}")

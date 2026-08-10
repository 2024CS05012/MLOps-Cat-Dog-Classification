import argparse
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.preprocess import preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(args.raw_dir, args.processed_dir)

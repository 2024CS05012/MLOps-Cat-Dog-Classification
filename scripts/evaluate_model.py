import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.api.inference import load_model
from src.config import MODEL_PATH, PROCESSED_DATA_DIR
from src.data.dataset import image_folder_dataset
from src.models.train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_path)
    dataset = image_folder_dataset(args.processed_dir, "test")
    loss, accuracy, _, _ = evaluate(model, DataLoader(dataset, batch_size=16), nn.CrossEntropyLoss())
    print({"test_loss": loss, "test_accuracy": accuracy})

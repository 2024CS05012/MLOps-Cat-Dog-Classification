import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, MODEL_DIR, PLOTS_DIR, PROCESSED_DATA_DIR
from src.data.dataset import image_folder_dataset
from src.models.model import SimpleCNN


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def save_curves(history: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("loss", "accuracy"):
        plt.figure()
        plt.plot(history[f"train_{metric}"], label=f"train_{metric}")
        plt.plot(history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png")
        plt.close()


def train_model(processed_dir: Path, epochs: int, batch_size: int, learning_rate: float) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = image_folder_dataset(processed_dir, "train")
    val_dataset = image_folder_dataset(processed_dir, "val")
    test_dataset = image_folder_dataset(processed_dir, "test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0
    best_path = MODEL_DIR / "best_model.pt"
    latest_path = MODEL_DIR / "latest_model.pt"
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    mlflow.set_experiment("cats-dogs-classification")
    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate})
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion)
            _, train_accuracy, _, _ = evaluate(model, train_loader, criterion)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_accuracy)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES}, best_path)

        torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES}, latest_path)
        test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, criterion)
        save_curves(history, PLOTS_DIR)
        matrix = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrix.png")
        plt.close()
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifacts(str(PLOTS_DIR))
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.processed_dir, args.epochs, args.batch_size, args.learning_rate)

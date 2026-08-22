from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from src.models.model import SimpleCNN


def preprocessing_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(model_path: Path = MODEL_PATH) -> SimpleCNN:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run `dvc pull` or `dvc repro` before starting the inference service."
        )
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_image(model: SimpleCNN, image: Image.Image) -> dict[str, object]:
    tensor = preprocessing_transform()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    scores = {class_name: float(probabilities[index]) for index, class_name in enumerate(CLASS_NAMES)}
    predicted_label = max(scores, key=scores.get)
    return {"label": predicted_label, "probabilities": scores}

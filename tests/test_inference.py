from PIL import Image

from src.api.inference import predict_image
from src.config import CLASS_NAMES
from src.models.model import SimpleCNN


def test_predict_image_returns_label_and_probabilities() -> None:
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.eval()
    image = Image.new("RGB", (224, 224), color=(120, 80, 40))

    prediction = predict_image(model, image)

    assert prediction["label"] in CLASS_NAMES
    assert set(prediction["probabilities"]) == set(CLASS_NAMES)
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5

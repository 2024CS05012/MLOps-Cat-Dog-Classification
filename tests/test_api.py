from fastapi.testclient import TestClient

from src.api import main as main_module


def test_health_endpoint() -> None:
    client = TestClient(main_module.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_without_model_returns_503() -> None:
    client = TestClient(main_module.app)
    original_model = main_module.model
    main_module.model = None
    try:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", b"not-a-real-image", "image/png")},
        )
    finally:
        main_module.model = original_model

    assert response.status_code == 503
    assert "Model is not available" in response.json()["detail"]

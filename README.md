# Cats vs Dogs MLOps Pipeline

End-to-end MLOps assignment project for binary image classification on the Kaggle Cats and Dogs dataset.

## Rubric Mapping

| Milestone | What is included |
| --- | --- |
| M1 Model development and tracking | PyTorch baseline CNN, DVC pipeline, MLflow logging, metrics, confusion matrix, saved model |
| M2 Packaging and containerization | FastAPI service, `/health`, `/predict`, pinned requirements, Dockerfile |
| M3 CI pipeline | Pytest tests, GitHub Actions CI, Docker build and optional registry push |
| M4 CD and deployment | Docker Compose, Kubernetes manifests, GitHub Actions CD, smoke test |
| M5 Monitoring and logs | Request logging middleware, in-app request/latency metrics, simulated post-deployment batch script |

## Project Layout

```text
.
├── .github/workflows/      # CI and CD pipelines
├── data/                   # DVC-tracked raw/processed/sample data
├── artifacts/              # model, plots, reports, predictions
├── src/                    # preprocessing, model training, API code
├── monitoring/             # logging and metrics helpers
├── deployment/             # docker-compose and Kubernetes manifests
├── scripts/                # dataset, train, evaluate, smoke, simulation scripts
├── tests/                  # pytest tests
├── Dockerfile
├── dvc.yaml
└── requirements.txt
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Dataset

Download the Kaggle Cats and Dogs dataset and place it in `data/raw/` with this structure:

```text
data/raw/
├── cat/
│   ├── cat_001.jpg
│   └── ...
└── dog/
    ├── dog_001.jpg
    └── ...
```

You can use the built-in Kaggle downloader in this project:

```bash
pip install -r requirements.txt
python scripts/download_kaggle_dataset.py --dataset-name bhavikjikadara/dog-and-cat-classification-dataset --output-dir data/raw
```

Keep your local Kaggle credentials outside the repository using the standard Kaggle JSON file:

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'JSON'
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
JSON
chmod 600 ~/.kaggle/kaggle.json
```

The repo ignores `.kaggle/` via `.gitignore`, so this file stays local only.

Alternatively, if you prefer the direct Kaggle API snippet:

```python
import kagglehub

path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Path to dataset files:", path)
```

Then version the raw image folders with DVC and run the reproducible pipeline:

```bash
dvc init
dvc add data/raw/cat data/raw/dog
dvc repro
```

The DVC pipeline preprocesses images to 224x224 RGB and splits them into train, validation, and test folders.

## Training

```bash
python3 scripts/train_model.py --epochs 3 --batch-size 16
```

Outputs:

- `artifacts/models/best_model.pt`
- `artifacts/models/latest_model.pt`
- `artifacts/plots/loss_curve.png`
- `artifacts/plots/accuracy_curve.png`
- `artifacts/plots/confusion_matrix.png`
- `mlruns/` MLflow experiment runs

## API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/sample/cat.jpg"
```

## Docker

```bash
docker build -t cats-dogs-mlops:latest .
docker run -p 8000:8000 cats-dogs-mlops:latest
```

## Deployment

Docker Compose:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

CI/CD deployment pulls the latest image published to GitHub Container Registry:

```bash
export IMAGE_NAME=ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY/cats-dogs-mlops:latest
docker pull "$IMAGE_NAME"
docker compose -f deployment/docker-compose.yml up -d --no-build
```

Kubernetes:

```bash
# Replace YOUR_GITHUB_USERNAME/YOUR_REPOSITORY in deployment/k8s/deployment.yaml first.
kubectl apply -f deployment/k8s/
```

## Smoke Test

```bash
python3 scripts/smoke_test.py --base-url http://localhost:8000 --image data/sample/cat.jpg
```

## Final Submission Checklist

- Commit source code and config files to Git.
- Commit DVC metadata, including `data/raw/cat.dvc`, `data/raw/dog.dvc`, `dvc.yaml`, and `dvc.lock`.
- Include trained model artifacts in the final zip, or include DVC remote/cache access instructions so `dvc pull` can restore them.
- Export a short screen recording showing code change, CI/CD run, deployment, health check, and prediction.

# MLOps Pipeline Report: Cats vs Dogs Binary Image Classification

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for a binary image classification use case: identifying whether an uploaded pet image is a cat or a dog. The solution is designed for a pet adoption platform where model predictions can support image tagging or listing verification.

The pipeline covers dataset preparation, model training, experiment tracking, artifact versioning, API packaging, containerization, CI/CD automation, deployment, smoke testing, and basic post-deployment monitoring.

## 2. Use Case and Dataset

The selected use case is binary image classification for cats and dogs.

Dataset:

- Source: Kaggle Cats and Dogs classification dataset
- Classes: `cat` and `dog`
- Raw data location: `data/raw/cat` and `data/raw/dog`
- DVC metadata: `data/raw/cat.dvc` and `data/raw/dog.dvc`

Preprocessing:

- Images are loaded as RGB.
- EXIF orientation is corrected.
- Images are resized to `224x224`.
- Data is split into train, validation, and test sets using an `80% / 10% / 10%` split.
- Processed data is stored under `data/processed`.

Data augmentation:

- Training data uses random horizontal flip.
- Training data uses random rotation.
- Training data uses color jitter.
- Validation and test data use deterministic transforms only.

Relevant files:

- `src/data/preprocess.py`
- `src/data/dataset.py`
- `scripts/preprocess_data.py`
- `dvc.yaml`
- `dvc.lock`

## 3. Tools and Technologies

The project uses the following open-source tools:

| Area | Tool |
| --- | --- |
| Source code versioning | Git |
| Data and artifact versioning | DVC |
| Model framework | PyTorch |
| Experiment tracking | MLflow |
| API service | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment target | Docker Compose |
| Optional deployment manifests | Kubernetes |
| Testing | Pytest |
| Monitoring/logging | Python logging and in-app counters |

## 4. M1: Model Development and Experiment Tracking

### 4.1 Data and Code Versioning

Source code is versioned using Git. The repository includes project source code, tests, scripts, Docker configuration, deployment manifests, CI/CD workflows, and setup documentation.

DVC is used to version the dataset and pipeline artifacts. The raw cat and dog folders are tracked using:

- `data/raw/cat.dvc`
- `data/raw/dog.dvc`

The DVC pipeline is defined in `dvc.yaml` and locked in `dvc.lock`.

DVC stages:

| Stage | Purpose |
| --- | --- |
| `preprocess` | Converts raw images into processed train/validation/test folders |
| `train` | Trains the baseline CNN and generates model/plot artifacts |

### 4.2 Model Building

The project implements a baseline convolutional neural network using PyTorch.

Model file:

- `src/models/model.py`

Training logic:

- `src/models/train.py`
- `scripts/train_model.py`

Saved model artifacts:

- `artifacts/models/best_model.pt`
- `artifacts/models/latest_model.pt`

The model is serialized using PyTorch checkpoint format (`.pt`). The checkpoint stores:

- model state dictionary
- class names

### 4.3 Experiment Tracking

MLflow is used for experiment tracking.

The training process logs:

- epochs
- batch size
- learning rate
- train loss
- validation loss
- train accuracy
- validation accuracy
- test loss
- test accuracy
- model checkpoint
- loss curve
- accuracy curve
- confusion matrix

Generated visual artifacts:

- `artifacts/plots/loss_curve.png`
- `artifacts/plots/accuracy_curve.png`
- `artifacts/plots/confusion_matrix.png`

## 5. M2: Model Packaging and Containerization

### 5.1 Inference Service

The trained model is wrapped in a FastAPI REST API.

Main API file:

- `src/api/main.py`

Inference utility:

- `src/api/inference.py`

API endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Returns service status and whether model is loaded |
| `/predict` | POST | Accepts an uploaded image and returns predicted label and class probabilities |
| `/metrics` | GET | Returns request count and average latency |

The prediction endpoint accepts an image file and returns a response containing:

- predicted label
- probability for `cat`
- probability for `dog`

### 5.2 Environment Specification

The project uses pinned dependencies for reproducibility.

Two dependency files are provided:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Full development, training, testing, DVC, MLflow, and CI environment |
| `requirements-api.txt` | Smaller inference-only environment for the Docker image |

This separation keeps the deployed container smaller while preserving the full training and experiment environment for development and CI.

### 5.3 Containerization

The inference service is containerized using Docker.

Docker file:

- `Dockerfile`

The Docker image:

- uses `python:3.11-slim`
- installs API dependencies from `requirements-api.txt`
- copies API source code and monitoring helpers
- copies model artifacts from `artifacts/models`
- verifies that `artifacts/models/best_model.pt` exists
- starts the FastAPI service with Uvicorn

Example local commands:

```bash
docker build -t cats-dogs-mlops:latest .
docker run -p 8000:8000 cats-dogs-mlops:latest
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/sample/cat.jpg"
```

## 6. M3: CI Pipeline for Build, Test, and Image Creation

### 6.1 Automated Testing

Automated tests are implemented using Pytest.

Test files:

- `tests/test_preprocess.py`
- `tests/test_inference.py`
- `tests/test_api.py`
- `tests/test_dataset_download.py`
- `tests/test_simulate_requests.py`

The tests cover:

- image preprocessing and resizing
- deterministic train/validation/test splitting
- inference utility output format
- API health endpoint
- API behavior when model artifact is unavailable
- simulated post-deployment request generation

Local test result:

```text
7 passed
```

### 6.2 CI Setup

GitHub Actions is used for CI.

Workflow file:

- `.github/workflows/ci.yml`

The CI workflow runs on:

- push to any branch
- pull request

CI steps:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs dependencies from `requirements.txt`.
4. Restores artifacts from DVC or rebuilds them using Kaggle credentials.
5. Verifies that `artifacts/models/best_model.pt` exists.
6. Runs unit tests with Pytest.
7. Builds the Docker image.
8. Pushes the image to GitHub Container Registry on push events.

### 6.3 Artifact Publishing

The CI pipeline publishes Docker images to GitHub Container Registry.

Image tags:

- commit SHA tag
- `latest` tag

The workflow lowercases the GitHub repository name before creating the GHCR image path.

## 7. M4: CD Pipeline and Deployment

### 7.1 Deployment Target

The main deployment target is Docker Compose.

Deployment file:

- `deployment/docker-compose.yml`

The Compose deployment:

- runs the API container
- exposes port `8000`
- supports using the image published by CI/CD through the `IMAGE_NAME` environment variable

Kubernetes manifests are also included as an optional deployment target:

- `deployment/k8s/deployment.yaml`
- `deployment/k8s/service.yaml`

### 7.2 CD Flow

GitHub Actions is used for CD.

Workflow file:

- `.github/workflows/cd.yml`

The CD workflow is triggered when the CI workflow completes successfully on the `main` branch.

CD steps:

1. Checks out the repository at the CI commit SHA.
2. Sets up Python.
3. Installs smoke test dependencies.
4. Restores DVC metadata or artifacts for smoke inputs.
5. Builds the deployment image name.
6. Logs in to GitHub Container Registry.
7. Pulls the latest Docker image.
8. Deploys the service using Docker Compose.
9. Waits for the API to start.
10. Runs smoke tests.
11. Collects post-deployment prediction results.
12. Uploads the post-deployment report as a workflow artifact.

### 7.3 Smoke Tests

Smoke testing is implemented in:

- `scripts/smoke_test.py`

The smoke test:

- calls `/health`
- sends one image to `/predict`
- fails if either request fails

Example command:

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

## 8. M5: Monitoring, Logs, and Post-Deployment Tracking

### 8.1 Basic Monitoring and Logging

The FastAPI service includes request logging middleware.

Relevant files:

- `src/api/main.py`
- `monitoring/logging_config.py`
- `monitoring/metrics.py`

Logged request details:

- request path
- HTTP method
- status code
- latency in milliseconds

Prediction logs include:

- uploaded filename
- predicted label

The API avoids logging raw image content.

The service also exposes basic runtime metrics through:

- `/metrics`

Metrics include:

- request count
- average latency in seconds

### 8.2 Model Performance Tracking After Deployment

Post-deployment request simulation is implemented in:

- `scripts/simulate_requests.py`

The script sends a small labeled batch to the deployed API and records:

- image name
- true label
- predicted label
- class probabilities

Example report:

- `artifacts/reports/post_deploy_predictions.example.json`

The CD workflow uploads the generated post-deployment prediction report as a GitHub Actions artifact named:

- `post-deployment-predictions`

## 9. Reproducibility Instructions

Create and activate the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Download the Kaggle dataset:

```bash
python scripts/download_kaggle_dataset.py \
  --dataset-name bhavikjikadara/dog-and-cat-classification-dataset \
  --output-dir data/raw
```

Run the DVC pipeline:

```bash
dvc repro
```

Run tests:

```bash
pytest -q
```

Run the API locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Run smoke test:

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

Run simulated post-deployment tracking:

```bash
python scripts/simulate_requests.py \
  --base-url http://localhost:8000 \
  --output artifacts/reports/post_deploy_predictions.json
```

## 10. Assignment Coverage Summary

| Assignment Requirement | Status | Evidence |
| --- | --- | --- |
| Git source code versioning | Covered | Repository source files and workflows |
| DVC dataset/preprocessed data tracking | Covered | `data/raw/*.dvc`, `dvc.yaml`, `dvc.lock` |
| Baseline model | Covered | `src/models/model.py` |
| Serialized trained model | Covered | `artifacts/models/best_model.pt` |
| Experiment tracking | Covered | MLflow logging in `src/models/train.py` |
| Confusion matrix and loss curves | Covered | `artifacts/plots` |
| REST inference service | Covered | FastAPI app in `src/api/main.py` |
| Health endpoint | Covered | `/health` |
| Prediction endpoint | Covered | `/predict` |
| Pinned dependencies | Covered | `requirements.txt`, `requirements-api.txt` |
| Dockerfile | Covered | `Dockerfile` |
| Unit tests | Covered | `tests/` |
| CI pipeline | Covered | `.github/workflows/ci.yml` |
| Docker image publishing | Covered | GHCR push in CI workflow |
| Deployment manifests | Covered | Docker Compose and Kubernetes YAML |
| CD pipeline | Covered | `.github/workflows/cd.yml` |
| Smoke test | Covered | `scripts/smoke_test.py` |
| Request/response logging | Covered | API middleware logging |
| Basic metrics | Covered | `/metrics` endpoint |
| Post-deployment prediction tracking | Covered | `scripts/simulate_requests.py` |
| ZIP submission | To be prepared | Final packaging step |
| Screen recording | To be prepared | Final demonstration step |

## 11. Notes and Limitations

Large files such as the full Kaggle dataset, processed image folders, local DVC cache, and trained model artifacts may be excluded from Git to keep the repository lightweight. The project includes setup instructions and CI fallback logic to regenerate required artifacts using Kaggle credentials.

Before final submission, ensure that either:

- the trained model artifact is included in the ZIP file, or
- the evaluator can regenerate it using the documented setup process.

For the screen recording, demonstrate the following flow:

1. Make a small code change.
2. Show CI running tests and building/pushing the Docker image.
3. Show CD deploying the image.
4. Call `/health`.
5. Call `/predict`.
6. Show the post-deployment prediction report or workflow artifact.

## 12. Conclusion

The project satisfies the core requirements of the assignment by implementing a complete MLOps workflow for binary image classification. It includes reproducible preprocessing and training, experiment tracking, containerized inference, automated testing, CI image creation and publishing, CD deployment, smoke testing, logging, basic metrics, and post-deployment prediction tracking.

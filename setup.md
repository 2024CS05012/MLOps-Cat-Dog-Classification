# Setup Guide for this MLOps Assignment

This project is intentionally structured so the assignment package stays small enough to upload under a 10 MB limit. The large files such as the Kaggle dataset, processed data, and trained model are downloaded or generated locally during setup and are not committed to the repository.

## 1) Requirements

- macOS or Linux
- Python 3.11
- Git
- Internet access for Kaggle download

## 2) Clone or download the project

```bash
cd /path/to/project
```

## 3) Create a Python environment

Use Python 3.11, because the pinned ML stack in this project depends on torch and torchvision versions that are compatible with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip
pip install -r requirements.txt
```

The Docker inference image intentionally uses the smaller `requirements-api.txt` file. Keep
`requirements.txt` for training, DVC, MLflow, testing, and CI jobs.

## 4) Configure Kaggle credentials

IMPORTANT: do not commit these credentials to Git.

### Recommended method: Kaggle JSON file

Create this file in your home directory:

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

Replace the placeholder values with your actual Kaggle username and API key.

## 5) Download the dataset

From the project root, run:

```bash
python scripts/download_kaggle_dataset.py \
  --dataset-name bhavikjikadara/dog-and-cat-classification-dataset \
  --output-dir data/raw
```

This script will:
- download the Kaggle dataset,
- discover the real `PetImages/Cat` and `PetImages/Dog` folders,
- copy them into `data/raw/cat` and `data/raw/dog`.

If you want to use the direct Kaggle API snippet instead:

```python
import kagglehub
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print(path)
```

## 6) Initialize DVC and preprocess the data

This repository already contains DVC metadata. If you are starting from a fresh zip without the `.dvc/`
directory, initialize DVC once:

```bash
dvc init
```

Track the raw dataset folders with DVC after downloading the dataset:

```bash
dvc add data/raw/cat data/raw/dog
```

Now preprocess the data:

```bash
python scripts/preprocess_data.py
```

This creates the processed train, validation, and test splits in:

```text
data/processed/
```

## 7) Train the model

```bash
python scripts/train_model.py --epochs 3 --batch-size 16
```

The trained model and plots are generated locally under the `artifacts/` folder.

This output is intentionally not committed in the assignment package, so the teacher or evaluator must regenerate it locally by running the above steps.

## 7.1) Configure artifact restoration for CI/CD

The Docker image must include `artifacts/models/best_model.pt`. Because large artifacts are
ignored by Git and no external DVC remote URL is used for this submission, CI regenerates
the artifacts from the public Kaggle dataset.

### Selected option: Kaggle secrets fallback

Add these GitHub repository secrets:

```text
KAGGLE_USERNAME
KAGGLE_KEY
```

The CI workflow uses those secrets to download the dataset and run `dvc repro` before running
tests and building the Docker image. The workflow fails clearly when neither a DVC remote nor
Kaggle credentials are available, because the Docker image must contain
`artifacts/models/best_model.pt`.

### Optional later: DVC remote

If you later get cloud/object storage, you can configure it as a DVC remote and push the
artifacts there. This is not required for the current submission because CI is configured
to regenerate artifacts from Kaggle credentials.

Before submission, verify the local DVC state is reproducible:

```bash
dvc repro
dvc status
```

`dvc status` should report that the pipeline is up to date. If it reports outputs that are
not in cache, run `dvc commit` after confirming the current generated artifacts are the
intended assignment artifacts. Run `dvc push` only if a DVC remote is configured.

## 8) Run tests

```bash
pytest -q
```

## 9) Run the API locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@data/raw/cat/9733.jpg"
```

If port 8000 is already occupied locally, use a different host port while keeping the container port fixed at 8000:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# in another terminal, or by binding via Docker:
docker run -p 8001:8000 cats-dogs-mlops:latest
curl http://localhost:8001/health
```

## 10) Smoke test

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

## 11) Docker deployment

```bash
docker build -t cats-dogs-mlops:latest .
docker run -p 8010:8000 cats-dogs-mlops:latest
```

Then test the service on:

```bash
curl http://localhost:8010/health
curl -X POST http://localhost:8010/predict -F "file=@data/raw/cat/9733.jpg"
```

For the CI/CD flow, GitHub Actions publishes the image to GitHub Container Registry and the CD workflow pulls that image before Docker Compose deployment.

## 12) Important for assignment submission

This repository is intentionally trimmed to stay under the assignment size limit. Large files such as the Kaggle dataset, processed data, local DVC cache, and trained model artifacts are not committed and must be regenerated during setup.

The expected reproducible workflow is:

1. Download repository code
2. Create environment
3. Add Kaggle credentials locally
4. Download dataset with Kaggle credentials
5. Track raw data metadata with DVC
6. Preprocess data
7. Train model locally
8. Run tests and smoke checks

If you are submitting a zip package, keep only the source/config files and any required DVC metadata. Do not include the local dataset cache or generated artifacts unless the assignment explicitly allows it.

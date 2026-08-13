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

## 6) Preprocess the data

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

The trained model and plots are saved under the `artifacts/` folder.

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
curl -X POST http://localhost:8000/predict -F "file=@data/sample/cat.jpg"
```

## 10) Smoke test

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
```

## 11) Docker deployment

```bash
docker build -t cats-dogs-mlops:latest .
docker run -p 8000:8000 cats-dogs-mlops:latest
```

## 12) Important for assignment submission

Do not upload the large generated files in the zip package. The expected workflow is:

1. Download repository code
2. Create environment
3. Add Kaggle credentials locally
4. Download dataset and train model locally
5. Run tests and smoke checks

This keeps the submission small and ensures the grader or evaluator can reproduce the project from source instructions alone.

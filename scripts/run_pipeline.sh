#!/usr/bin/env bash
set -euo pipefail

python3 scripts/preprocess_data.py
python3 scripts/train_model.py --epochs "${EPOCHS:-3}" --batch-size "${BATCH_SIZE:-16}"
pytest

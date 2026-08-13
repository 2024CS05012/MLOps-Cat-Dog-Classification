import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import parse_args, train_model


if __name__ == "__main__":
    args = parse_args()
    train_model(
        args.processed_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.max_train_samples,
        args.max_eval_samples,
    )

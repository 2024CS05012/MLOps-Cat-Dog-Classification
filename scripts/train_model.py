from src.models.train import parse_args, train_model


if __name__ == "__main__":
    args = parse_args()
    train_model(args.processed_dir, args.epochs, args.batch_size, args.learning_rate)

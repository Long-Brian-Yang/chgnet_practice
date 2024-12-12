import warnings
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import os
import csv
from datetime import datetime
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from dataset import load_dataset


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging system

    Args:
        log_dir (str): log directory

    Returns:
        None
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/finetuning.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """
    Parse command line arguments

    Args:
        None

    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser(description='CHGNet Finetuning')
    parser.add_argument('--json-path', type=str, default="./data/dataset.json",
                        help='Path to the JSON dataset file')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Proportion of dataset for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Proportion of dataset for validation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                        help='Learning rate for optimizer')
    parser.add_argument('--output-dir', type=str, default='./finetuning_results',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for computation (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def setup_csv_logger(output_dir: Path) -> tuple[Path, Path]:
    """
    Set up CSV logger for training metrics

    Args:
        output_dir: Directory to save logs

    Returns:
        tuple: (log_path, log_file)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = output_dir / "logs" / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "finetuning_log.csv"

    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    return log_path, log_file


def log_metrics(log_file: Path, epoch: int, train_loss: float, val_loss: float) -> None:
    """
    Log training metrics to CSV file

    Args:
        log_file: Path to log file
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
    """
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss])


def freeze_model_layers(model: CHGNet) -> CHGNet:
    """
    Freeze specific layers in the model for finetuning

    Args:
        model: CHGNet model

    Returns:
        CHGNet: Model with frozen layers
    """
    frozen_layers = [
        model.atom_embedding,
        model.bond_embedding,
        model.angle_embedding,
        model.bond_basis_expansion,
        model.angle_basis_expansion,
        model.atom_conv_layers[:-1],
        model.bond_conv_layers,
        model.angle_layers,
    ]

    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False

    return model


def plot_training_curves(history: dict, output_dir: Path, logger: logging.Logger) -> None:
    """
    Plot and save training curves

    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plot
        logger: Logger instance
    """
    plt.figure(figsize=(12, 8))
    fontsize = 24

    plt.plot(history["train"], label="Train Loss", marker="o")
    plt.plot(history["val"], label="Validation Loss", marker="s")

    plt.title("CHGNet Finetuning Progress", fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize-4)
    plt.ylabel("Loss", fontsize=fontsize-4)
    plt.tick_params(labelsize=fontsize-4)
    plt.legend(fontsize=fontsize-4)
    plt.grid(True)
    plt.tight_layout()

    plot_path = output_dir / "finetuning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Training curves saved to {plot_path}")


def run_finetuning(args) -> None:
    """
    Run CHGNet finetuning process

    Args:
        args (argparse.Namespace): command line arguments
    """
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "logs"))
        logger = logging.getLogger(__name__)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        log_path, log_file = setup_csv_logger(output_dir)
        logger.info(f"Logs will be saved to {log_path}")

        # Load dataset
        logger.info("Loading dataset...")
        train_loader, val_loader, test_loader = load_dataset(
            json_path=args.json_path,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        logger.info("Dataset loaded successfully")

        # Initialize and freeze model
        logger.info("Initializing model...")
        model = CHGNet()
        model = freeze_model_layers(model)
        logger.info("Model layers frozen for finetuning")

        # Setup trainer
        trainer = Trainer(
            model=model,
            targets="ef",
            optimizer="Adam",
            scheduler="CosLR",
            criterion="MSE",
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_device=args.device,
            print_freq=1
        )

        # Training loop
        logger.info("Starting finetuning...")
        train_losses = []
        val_losses = []

        for epoch in range(args.epochs):
            train_loss = trainer._train(train_loader, epoch)["e"]
            val_loss = trainer._validate(val_loader)["e"]

            log_metrics(log_file, epoch, train_loss, val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(f"Epoch {epoch + 1}/{args.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")

        # Save model
        model_path = output_dir / "checkpoints" / "chgnet_finetuned.pth"
        model_path.parent.mkdir(exist_ok=True)
        torch.save(trainer.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Plot results
        history = {"train": train_losses, "val": val_losses}
        plot_training_curves(history, output_dir, logger)

        logger.info("Finetuning completed successfully")

    except Exception as e:
        logger.error(f"Finetuning failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        run_finetuning(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")

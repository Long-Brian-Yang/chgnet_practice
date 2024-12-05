import os
import torch
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from dataset import load_dataset
import matplotlib.pyplot as plt
import csv
from datetime import datetime


def setup_logger(log_dir="logs", log_name="CHGNet_finetuning"):
    """Set up a CSV logger to record training progress."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, log_name, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "training_log.csv")
    
    # Write the header
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
    
    return log_path, log_file


def log_epoch_results(log_file, epoch, train_loss, val_loss):
    """Append epoch results to the CSV log file."""
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss])

def freeze_model_layers(model):
    """Freeze the weights of specific layers in the CHGNet model."""
    for layer in [
        model.atom_embedding,
        model.bond_embedding,
        model.angle_embedding,
        model.bond_basis_expansion,
        model.angle_basis_expansion,
        model.atom_conv_layers[:-1],
        model.bond_conv_layers,
        model.angle_layers,
    ]:
        for param in layer.parameters():
            param.requires_grad = False

    return model

def train_model(
    json_path, batch_size, train_ratio, val_ratio, epochs, learning_rate, save_path, log_dir="logs"
):
    """
    Train a CHGNet model and log the results.
    
    Args:
        json_path (str): Path to the JSON dataset file.
        batch_size (int): Batch size for training.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        save_path (str): Path to save the trained model.
        log_dir (str): Directory to save the training logs.

    Returns:
        tuple: (training_history, log_path)
    """
    # Set up logger
    log_path, log_file = setup_logger(log_dir=log_dir)

    # Load dataset
    train_loader, val_loader, test_loader = load_dataset(
        json_path=json_path, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio
    )

    # Initialize CHGNet model
    chgnet = CHGNet()

    # Optionally freeze certain layers
    chgnet = freeze_model_layers(chgnet)

    # Define Trainer
    trainer = Trainer(
        model=chgnet,
        targets="ef",
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=epochs,
        learning_rate=learning_rate,
        use_device="cpu",
        print_freq=1,
    )

    # Train model and log results
    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train and validate for one epoch
        train_loss = trainer._train(train_loader, epoch)["e"]
        val_loss = trainer._validate(val_loader)["e"]

        # Log epoch results
        log_epoch_results(log_file, epoch, train_loss, val_loss)

        # Append to loss lists for return
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Save model
    checkpoint_dir = os.path.dirname(save_path)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
    torch.save(trainer.model.state_dict(), save_path)
    print(f"Training completed and model saved as {save_path}")

    return {"train": train_losses, "val": val_losses}, log_path

def plot_loss_curve(training_history, log_path):
    """
    Plot training and validation loss curves.

    Args:
        training_history (dict): A dictionary containing training and validation loss.
        log_path (str): Path to save the loss curve plot.
    """
    train_losses = training_history["train"]
    val_losses = training_history["val"]

    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")

    # Set plot title and labels
    plt.title("Training and Validation Loss vs Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(log_path, "finetuning_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.show()

    print(f"Loss curves saved as {loss_plot_path}")


if __name__ == "__main__":
    # Define parameters
    JSON_PATH = "vasp_dataset.json"
    BATCH_SIZE = 2
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    EPOCHS = 50
    LEARNING_RATE = 1e-2
    SAVE_MODEL_PATH = "logs/CHGNet_finetuning/checkpoints/chgnet_finetuned_model.pth"
    LOG_DIR = "logs"

    # Train and validate with logging
    training_history, log_path = train_model(
        json_path=JSON_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_path=SAVE_MODEL_PATH,
        log_dir=LOG_DIR,
    )

    # Plot loss curves
    plot_loss_curve(training_history, log_path)


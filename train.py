import torch
import torch.nn as nn
import numpy as np
from dataset import WallDataset, create_wall_dataloader
from jepa_models import RecurrentJEPA
from tqdm import tqdm


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds


def compute_vicreg_loss(
    predictions, targets, sim_weight=25.0, var_weight=25.0, cov_weight=1.0
):
    # Reshape from [batch_size, seq_len, embed_dim] to [batch_size * seq_len, embed_dim]
    predictions = predictions.reshape(-1, predictions.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])

    # Invariance loss (similar to MSE)
    sim_loss = nn.MSELoss()(predictions, targets)

    # Variance loss
    std_p = torch.sqrt(predictions.var(dim=0) + 1e-04)
    std_loss = torch.mean(torch.relu(1 - std_p))

    # Covariance loss
    predictions = predictions - predictions.mean(dim=0)
    cov_p = (predictions.T @ predictions) / (predictions.shape[0] - 1)
    cov_loss = off_diagonal(cov_p).pow_(2).sum() / predictions.shape[1]

    # Combine losses
    loss = sim_weight * sim_loss + var_weight * std_loss + cov_weight * cov_loss

    return loss


def off_diagonal(x):
    """Return off-diagonal elements of a square matrix."""
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train_model(
    model,
    dataloader,
    optimizer,
    epochs,
    device,
    patience=5,
    save_path="model_weights.pth",
):
    model = model.to(device)
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for e in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {e+1}/{epochs}")

        for i, batch in pbar:
            states, locations, actions = batch
            states, locations, actions = (
                states.to(device),
                locations.to(device),
                actions.to(device),
            )

            # Forward pass
            predictions, targets = model(states, actions)
            loss = compute_vicreg_loss(predictions, targets)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update target encoder with momentum
            model._momentum_update_target_encoder()

            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {e + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            best_epoch = e
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best loss: {best_loss:.4f}. Model weights saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement for {patience_counter} epochs. Best loss: {best_loss:.4f}"
            )

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after epoch {e+1}. Best epoch was {best_epoch+1}"
                )
                break


def main():
    save_path = "model_weights.pth"
    # Hyperparameters
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 20
    embed_dim = 768
    momentum = 0.99
    patience = 5

    # Define data, model, and optimizer
    device = get_device()
    model = RecurrentJEPA(embed_dim=embed_dim, momentum=momentum)
    train_dataloader = load_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    train_model(
        model,
        train_dataloader,
        optimizer,
        epochs,
        device,
        patience=patience,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import create_wall_dataloader
from JEPA_model import JEPAModel
import numpy as np
from tqdm import tqdm


def off_diagonal(x):
    """Return off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, sim_coef=25.0, std_coef=50.0, cov_coef=2.0):
    """Stronger regularization to prevent collapse"""
    B, T, D = z1.shape

    total_loss = 0
    for t in range(T):
        # Take each timestep: [B, D]
        z1_t = z1[:, t]
        z2_t = z2[:, t]

        # Stronger invariance loss
        sim_loss = F.mse_loss(z1_t, z2_t)

        # Stronger variance loss - crucial for preventing collapse
        std_z1 = torch.sqrt(z1_t.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2_t.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Stronger covariance loss - prevents dimensional collapse
        z1_t = z1_t - z1_t.mean(dim=0)
        z2_t = z2_t - z2_t.mean(dim=0)
        cov_z1 = (z1_t.T @ z1_t) / (z1_t.shape[0] - 1)
        cov_z2 = (z2_t.T @ z2_t) / (z2_t.shape[0] - 1)
        cov_loss = (
            off_diagonal(cov_z1).pow_(2).sum() / D
            + off_diagonal(cov_z2).pow_(2).sum() / D
        )

        # Stronger decorrelation between samples
        z1_norm = F.normalize(z1_t, dim=1)
        z2_norm = F.normalize(z2_t, dim=1)
        similarity_matrix = torch.mm(z1_norm, z2_norm.T)
        decorr_loss = torch.mean(
            torch.abs(similarity_matrix - torch.eye(z1.shape[0], device=z1.device))
        )

        loss = (
            sim_coef * sim_loss
            + std_coef * std_loss
            + cov_coef * cov_loss
            + decorr_loss
        )
        total_loss += loss

    return total_loss / T


def monitor_collapse(z1, epoch):
    """Monitor for signs of collapse"""
    with torch.no_grad():
        # Check variance across batch
        var_per_dim = z1.var(dim=0)
        active_dims = (var_per_dim > 0.1).sum()
        mean_var = var_per_dim.mean()

        print(f"Epoch {epoch}")
        print(f"Active dimensions: {active_dims}/{z1.shape[1]}")
        print(f"Mean variance: {mean_var:.4f}")

        # Check for similar representations
        z1_norm = F.normalize(z1, dim=1)
        similarity = torch.mm(z1_norm, z1_norm.T)
        mean_sim = similarity.mean()
        print(f"Mean similarity between samples: {mean_sim:.4f}")


def train_jepa(
    model,
    train_loader,
    optimizer,
    device,
    epochs=100,
    log_interval=10,
    patience=4,
    min_delta=1e-4,
):
    model.train()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Properly configure the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True, threshold=min_delta
    )

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            optimizer.zero_grad()

            predictions, targets = model(states, actions)
            loss = vicreg_loss(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.momentum_update()

            total_loss += loss.item()
            if batch_idx % log_interval == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
                    }
                )

            # Monitor every N batches
            if batch_idx % 100 == 0:
                monitor_collapse(predictions[:, 0], epoch)  # Monitor first timestep

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "model_weights.pth")
            print(f"New best model saved with loss: {avg_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            model.load_state_dict(best_model_state)
            break

        # Update learning rate
        scheduler.step(avg_loss)

    return model, best_loss


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train", batch_size=BATCH_SIZE, train=True
    )

    # Initialize model
    model = JEPAModel(latent_dim=256, momentum=0.99).to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train with early stopping
    model, best_loss = train_jepa(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        patience=4,  # Stop if no improvement for 4 epochs
        min_delta=1e-4,  # Minimum improvement threshold
    )

    print(f"Training finished with best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

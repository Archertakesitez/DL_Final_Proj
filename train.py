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


def vicreg_loss(z1, z2, sim_coef=25.0, std_coef=25.0, cov_coef=1.0):
    """VicReg loss computation per timestep"""
    B, T, D = z1.shape

    total_loss = 0
    for t in range(T):
        # Take each timestep: [B, D]
        z1_t = z1[:, t]
        z2_t = z2[:, t]

        # Invariance loss
        sim_loss = F.mse_loss(z1_t, z2_t)

        # Variance loss
        std_z1 = torch.sqrt(z1_t.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2_t.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss
        z1_t = z1_t - z1_t.mean(dim=0)
        z2_t = z2_t - z2_t.mean(dim=0)
        cov_z1 = (z1_t.T @ z1_t) / (z1_t.shape[0] - 1)
        cov_z2 = (z2_t.T @ z2_t) / (z2_t.shape[0] - 1)
        cov_loss = (
            off_diagonal(cov_z1).pow_(2).sum() / D
            + off_diagonal(cov_z2).pow_(2).sum() / D
        )

        loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss
        total_loss += loss

    return total_loss / T  # Average over timesteps


def train_jepa(
    model,
    train_loader,
    optimizer,
    device,
    epochs=100,
    log_interval=10,
    patience=4,  # Number of epochs to wait for improvement
    min_delta=1e-4,  # Minimum change to qualify as an improvement
):
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states.to(device)  # [B, T, C, H, W]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            optimizer.zero_grad()

            # Get predictions and targets in one forward pass
            predictions, targets = model(states, actions)

            # Compute VICReg loss
            loss = vicreg_loss(predictions, targets)

            # Monitor for collapse
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    pred_std = predictions.std().item()
                    target_std = targets.std().item()
                    print(f"Pred std: {pred_std:.4f}, Target std: {target_std:.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.momentum_update()


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train", batch_size=BATCH_SIZE, train=True
    )

    # Initialize model
    model = JEPAModel(latent_dim=256, use_momentum=True).to(DEVICE)

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

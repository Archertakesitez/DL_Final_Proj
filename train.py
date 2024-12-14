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


def vicreg_loss(z1, z2, sim_coef=100.0, std_coef=10.0, cov_coef=1.0, target_std=0.3):
    """
    VicReg loss with stronger variance stabilization
    """
    B, T, D = z1.shape

    total_loss = 0
    for t in range(1, T):
        z1_t = z1[:, t]
        z2_t = z2[:, t]

        # Invariance loss
        sim_loss = F.mse_loss(z1_t, z2_t)

        # Stronger variance stabilization
        std_z1 = torch.sqrt(z1_t.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2_t.var(dim=0) + 1e-04)

        # Quadratic penalty for deviating from target_std
        std_loss = ((std_z1 - target_std).pow(2)).mean() + (
            (std_z2 - target_std).pow(2)
        ).mean()

        # Covariance loss
        z1_t = z1_t - z1_t.mean(dim=0)
        z2_t = z2_t - z2_t.mean(dim=0)
        cov_z1 = (z1_t.T @ z1_t) / (B - 1)
        cov_z2 = (z2_t.T @ z2_t) / (B - 1)
        cov_loss = (
            off_diagonal(cov_z1).pow_(2).sum() / D
            + off_diagonal(cov_z2).pow_(2).sum() / D
        )

        loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss
        total_loss += loss

    return total_loss / (T - 1)


def apply_trajectory_augmentations(states, actions, p=0.3):
    """
    Apply augmentations that preserve action-state relationships
    states: [B, T, C, H, W]
    actions: [B, T-1, 2] (delta_x, delta_y)
    """
    B, T, C, H, W = states.shape

    # Horizontal flip - need to flip both states AND actions
    if torch.rand(1) < p:
        states = torch.flip(states, dims=[-1])
        actions[:, :, 0] = -actions[:, :, 0]  # Flip x-direction actions

    # 90-degree rotations
    if torch.rand(1) < p:
        k = torch.randint(1, 4, (1,)).item()  # Convert tensor to integer with .item()
        states = torch.rot90(states, k=k, dims=[-2, -1])

        # Rotate actions by k*90 degrees
        for _ in range(k):
            actions_x = actions[:, :, 0].clone()
            actions[:, :, 0] = -actions[:, :, 1]  # x = -y
            actions[:, :, 1] = actions_x  # y = x

    return states, actions


def train_jepa(
    model,
    train_loader,
    optimizer,
    device,
    epochs=100,
    log_interval=100,
    patience=4,  # Number of epochs to wait for improvement
    min_delta=1e-4,  # Minimum change to qualify as an improvement
):
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True, threshold=min_delta
    )

    STD_COLLAPSE_THRESHOLD = 0.05

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            # Apply trajectory-consistent augmentations
            states, actions = apply_trajectory_augmentations(states, actions)

            optimizer.zero_grad()
            # Only pass initial states and full action sequence
            init_states = states[:, 0:1]  # Take only first timestep [B, 1, C, H, W]
            predictions, targets = model(states=init_states, actions=actions)
            loss = vicreg_loss(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.momentum_update()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss/(batch_idx+1):.4f}",
                }
            )

            # Collapse monitoring (in a separate line)
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    pred_mean = predictions.mean().item()
                    pred_std = predictions.std().item()
                    target_mean = targets.mean().item()
                    target_std = targets.std().item()

                    # Use tqdm.write to print without breaking the progress bar
                    tqdm.write("\nCollapse Monitoring:")
                    tqdm.write(f"Batch {batch_idx}")
                    tqdm.write(f"Pred   μ/σ: {pred_mean:.4f}/{pred_std:.4f}")
                    tqdm.write(f"Target μ/σ: {target_mean:.4f}/{target_std:.4f}")
                    if pred_std < STD_COLLAPSE_THRESHOLD:
                        tqdm.write("⚠️  WARNING: Possible collapse detected!")
                    tqdm.write("")  # Empty line for spacing

        # End of epoch summary
        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"\nEpoch {epoch+1} Summary:")
        tqdm.write(f"Average Loss: {avg_loss:.4f}")
        tqdm.write(f"Final Pred   μ/σ: {pred_mean:.4f}/{pred_std:.4f}")
        tqdm.write(f"Final Target μ/σ: {target_mean:.4f}/{target_std:.4f}")
        if pred_std < STD_COLLAPSE_THRESHOLD:
            tqdm.write("⚠️  WARNING: Epoch ended with possible collapse!")
        tqdm.write("-" * 50)

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "model_weights.pth")
            tqdm.write(f"New best model saved with loss: {avg_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            tqdm.write(f"Early stopping triggered after {epoch + 1} epochs")
            model.load_state_dict(best_model_state)
            break

        scheduler.step(avg_loss)

    return model, best_loss


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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import create_wall_dataloader
from JEPA_model import JEPAModel
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def byol_loss(predictions, targets, temperature=0.1):
    """BYOL loss: negative cosine similarity"""
    B, T, D = predictions.shape
    total_loss = 0

    for t in range(T):
        pred_norm = F.normalize(predictions[:, t], dim=1)
        target_norm = F.normalize(targets[:, t], dim=1)
        similarity = torch.einsum("bd,bd->b", pred_norm, target_norm) / temperature
        loss = -similarity.mean()
        total_loss += loss

    return total_loss / T


def visualize_batch(states, actions, batch_idx, epoch, writer, max_samples=8):
    """
    Visualize a batch of states and save to TensorBoard
    Args:
        states: [B, T, 2, H, W] tensor of states
        actions: [B, T-1, 2] tensor of actions
        batch_idx: current batch index
        epoch: current epoch
        writer: TensorBoard writer
        max_samples: maximum number of samples to visualize
    """
    B, T, C, H, W = states.shape
    samples = min(B, max_samples)
    
    # Create figure for each timestep
    for t in range(T):
        fig, axes = plt.subplots(samples, 2, figsize=(8, 2*samples))
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}, Timestep {t}')
        
        for i in range(samples):
            # Plot agent channel
            axes[i, 0].imshow(states[i, t, 0].cpu(), cmap='gray')
            axes[i, 0].set_title('Agent Position')
            axes[i, 0].axis('off')
            
            # Plot walls channel
            axes[i, 1].imshow(states[i, t, 1].cpu(), cmap='gray')
            axes[i, 1].set_title('Walls & Borders')
            axes[i, 1].axis('off')
            
            # Add action arrow if not last timestep
            if t < T-1:
                action = actions[i, t].cpu()
                axes[i, 0].arrow(W/2, H/2, action[0]*10, action[1]*10,
                               head_width=2, head_length=2, fc='r', ec='r')
        
        # Log to TensorBoard
        writer.add_figure(f'Samples/timestep_{t}', fig, epoch)
        plt.close(fig)


def train_jepa(
    model,
    train_loader,
    val_loader,  # Added validation loader
    optimizer,
    device,
    epochs=100,
    log_interval=10,
    patience=4,
    min_delta=1e-4,
    warmup_epochs=5,  # Added warmup epochs
):
    model.train()
    writer = SummaryWriter('runs/jepa_experiment')  # TensorBoard logging
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Calculate warmup steps and total steps
    num_warmup_steps = warmup_epochs * len(train_loader)
    num_training_steps = epochs * len(train_loader)
    
    # Initialize scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps,
        num_training_steps
    )

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            states = batch.states.to(device)
            actions = batch.actions.to(device)

            # Data augmentation (random horizontal flip)
            if random.random() > 0.5:
                states = torch.flip(states, dims=[3])  # Flip horizontally
                actions[:, :, 0] = -actions[:, :, 0]  # Flip x-direction actions

            optimizer.zero_grad()

            predictions = model(states, actions)
            targets = model.compute_target(states)
            loss = byol_loss(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.momentum_update()
            scheduler.step()

            total_train_loss += loss.item()
            
            # Logging
            if batch_idx % log_interval == 0:
                writer.add_scalar('Loss/train_step', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_train_loss/(batch_idx+1):.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })

            # Visualize every N batches
            if batch_idx % 100 == 0:  # Adjust frequency as needed
                visualize_batch(states, actions, batch_idx, epoch, writer)

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch.states.to(device)
                actions = batch.actions.to(device)
                
                predictions = model(states, actions)
                targets = model.compute_target(states)
                loss = byol_loss(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, "model_weights.pth")
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            model.load_state_dict(best_model_state)
            break

    writer.close()
    return model, best_val_loss


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        batch_size=BATCH_SIZE,
        train=True
    )
    
    # Create validation loader using a portion of training data
    val_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/val",
        batch_size=BATCH_SIZE,
        train=False
    )

    # Initialize model
    model = JEPAModel(latent_dim=256, use_momentum=True).to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Train with early stopping
    model, best_loss = train_jepa(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        patience=4,
        min_delta=1e-4,
        warmup_epochs=5
    )

    print(f"Training finished with best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

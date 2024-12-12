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


def variance_loss(embeddings, eps=1e-4):
    """
    Implements VICReg-style variance loss to ensure each dimension has high variance.
    
    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, embed_dim)
        eps (float): Small constant for numerical stability
    """
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.mean(torch.relu(1 - std))


def covariance_loss(embeddings):
    """
    Implements VICReg-style covariance loss to decorrelate different dimensions.
    
    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, embed_dim)
    """
    batch_size, embed_dim = embeddings.size()
    
    # Center the embeddings
    embeddings = embeddings - embeddings.mean(dim=0)
    
    # Compute covariance matrix
    cov = (embeddings.T @ embeddings) / (batch_size - 1)
    
    # Zero out diagonal
    diagonal_mask = ~torch.eye(embed_dim, device=embeddings.device).bool()
    cov_off_diag = cov[diagonal_mask].pow(2)
    
    return cov_off_diag.sum() / embed_dim


def barlow_twins_loss(pred_embeddings, target_embeddings, lambda_param=0.0051):
    """
    Implements Barlow Twins cross-correlation loss.
    
    Args:
        pred_embeddings (torch.Tensor): Predicted embeddings (batch_size, embed_dim)
        target_embeddings (torch.Tensor): Target embeddings (batch_size, embed_dim)
        lambda_param (float): Off-diagonal scaling parameter
    """
    batch_size = pred_embeddings.size(0)
    
    # Normalize embeddings along batch dimension
    pred_norm = (pred_embeddings - pred_embeddings.mean(0)) / pred_embeddings.std(0)
    target_norm = (target_embeddings - target_embeddings.mean(0)) / target_embeddings.std(0)
    
    # Cross-correlation matrix
    c = torch.mm(pred_norm.T, target_norm) / batch_size
    
    # Loss computation
    on_diag = torch.diagonal(c).add_(-1).pow_(2)
    off_diag = c.flatten()[1:].view(c.size(0) - 1, c.size(1) + 1)[:, :-1].flatten()
    off_diag = off_diag.pow_(2).mul_(lambda_param)
    
    return on_diag.sum() + off_diag.sum()


def compute_loss(predictions, targets, reg_weight=0.1, bt_weight=0.1):
    """
    Computes total loss including MSE, VICReg components, and Barlow Twins loss.
    
    Args:
        predictions (torch.Tensor): Predicted embeddings
        targets (torch.Tensor): Target embeddings
        reg_weight (float): Weight for VICReg losses
        bt_weight (float): Weight for Barlow Twins loss
    """
    # Reshape if needed (batch_size * seq_len, embed_dim)
    if len(predictions.shape) > 2:
        pred_flat = predictions.reshape(-1, predictions.size(-1))
        target_flat = targets.reshape(-1, targets.size(-1))
    else:
        pred_flat = predictions
        target_flat = targets

    # Basic reconstruction loss
    mse = nn.MSELoss()(predictions, targets)
    
    # VICReg components
    var_loss = variance_loss(pred_flat) + variance_loss(target_flat)
    cov_loss = covariance_loss(pred_flat) + covariance_loss(target_flat)
    
    # Barlow Twins loss
    bt_loss = barlow_twins_loss(pred_flat, target_flat)
    
    # Combine all losses
    total_loss = mse + reg_weight * (var_loss + cov_loss) + bt_weight * bt_loss
    
    return total_loss


def train_model(
    model, dataloader, optimizer, epochs, device, save_path="pretrained_jepa_model.pth"
):
    model = model.to(device)
    best_loss = float("inf")

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

            # Compute loss
            loss = compute_loss(predictions, targets, reg_weight=0.1, bt_weight=0.005)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {e + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {save_path}.")


def main():
    save_path = "pretrained_jepa_model.pth"
    # Hyperparameters
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 10
    embed_dim = 768

    # Define data, model, and optimizer
    device = get_device()
    model = RecurrentJEPA(embed_dim=embed_dim)
    train_dataloader = load_data(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    train_model(model, train_dataloader, optimizer, epochs, device, save_path)


if __name__ == "__main__":
    main()

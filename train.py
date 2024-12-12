import torch
import torch.nn as nn
import numpy as np
from dataset import create_wall_dataloader
from jepa_models import RecurrentJEPA
from tqdm import tqdm


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"
    return create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )


def variance_loss(embeddings, eps=1e-4):
    """VICReg variance loss: ensures each dimension has high variance"""
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.mean(torch.relu(1 - std))


def covariance_loss(embeddings):
    """VICReg covariance loss: decorrelates different dimensions"""
    batch_size = embeddings.size(0)
    embeddings = embeddings - embeddings.mean(dim=0)
    cov = (embeddings.T @ embeddings) / (batch_size - 1)
    
    diagonal_mask = ~torch.eye(embeddings.size(1), device=embeddings.device).bool()
    cov_off_diag = cov[diagonal_mask].pow(2)
    
    return cov_off_diag.sum() / embeddings.size(1)


def barlow_twins_loss(pred_embeddings, target_embeddings, lambda_param=0.0051):
    """Barlow Twins loss: maintains invariance while preventing collapse"""
    batch_size = pred_embeddings.size(0)
    
    # Normalize embeddings
    pred_norm = (pred_embeddings - pred_embeddings.mean(0)) / pred_embeddings.std(0)
    target_norm = (target_embeddings - target_embeddings.mean(0)) / target_embeddings.std(0)
    
    # Cross-correlation matrix
    c = torch.mm(pred_norm.T, target_norm) / batch_size
    
    # Loss computation
    on_diag = torch.diagonal(c).add_(-1).pow_(2)
    off_diag = c.flatten()[1:].view(c.size(0) - 1, c.size(1) + 1)[:, :-1].flatten()
    off_diag = off_diag.pow_(2).mul_(lambda_param)
    
    return on_diag.sum() + off_diag.sum()


def compute_loss(predictions, targets):
    """
    Compute JEPA loss: F(τ) = ∑D(s̃ₙ, s'ₙ) + regularization
    """
    # Reshape if needed
    pred_flat = predictions.reshape(-1, predictions.size(-1))
    target_flat = targets.reshape(-1, targets.size(-1))

    # Basic distance in representation space
    mse = nn.MSELoss()(predictions, targets)
    
    # VICReg components to prevent collapse
    var_loss = variance_loss(pred_flat) + variance_loss(target_flat)
    cov_loss = covariance_loss(pred_flat) + covariance_loss(target_flat)
    
    # Barlow Twins component to prevent collapse
    bt_loss = barlow_twins_loss(pred_flat, target_flat)
    
    # Total loss with careful weighting
    total_loss = mse + 0.1 * (var_loss + cov_loss) + 0.005 * bt_loss
    
    return total_loss


def train_model(model, dataloader, optimizer, scheduler, epochs, device, save_path):
    model = model.to(device)
    best_loss = float("inf")
    
    for e in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {e+1}/{epochs}")
        
        for i, batch in pbar:
            states, locations, actions = batch
            
            # Forward pass
            predictions, targets = model(states, actions)
            
            # Compute loss
            loss = compute_loss(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            pbar.set_postfix({"Loss": loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {e + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {save_path}.")


def main():
    # Hyperparameters
    lr = 1e-4
    weight_decay = 1e-6
    epochs = 100
    embed_dim = 768
    
    device = get_device()
    model = RecurrentJEPA(embed_dim=embed_dim)
    train_dataloader = load_data(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    train_model(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        save_path="model_weights.pth"
    )


if __name__ == "__main__":
    main()

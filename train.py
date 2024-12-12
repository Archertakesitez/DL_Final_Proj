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


def variance_regularization(embeddings, eps=1e-4):
    """
    Computes variance regularization loss for embeddings.

    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, trajectory_length, embed_dim).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Variance regularization loss.
    """

    # Reshape to (batch_size * trajectory_length, emb_size)
    flattened = embeddings.view(-1, embeddings.size(-1))
    
    variance = torch.sqrt(torch.var(flattened, dim=0) + eps)
    reg_loss = torch.mean(torch.relu(1 - variance))
    
    return reg_loss


def covariance_regularization(embeddings):
    """
    Computes covariance regularization loss for embeddings.

    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, trajectory_length, embed_dim).

    Returns:
        torch.Tensor: Covariance regularization loss.
    """
    
    # Reshape to (batch_size * trajectory_length, emb_size)
    flattened = embeddings.view(-1, embeddings.size(-1))
    
    flattened = flattened - flattened.mean(dim=0)
    cov_matrix = (flattened.T @ flattened) / flattened.size(0)
    off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
    reg_loss = torch.sum(off_diag ** 2)

    return reg_loss


def contrastive_loss(predictions, targets, temperature=0.1):
    """
    Contrastive loss to ensure diversity among embeddings.

    Args:
        predictions (torch.Tensor): Predicted embeddings (batch_size, trajectory_length, embed_dim).
        targets (torch.Tensor): Target embeddings (batch_size, embed_dim).
        temperature (float): Temperature for scaling logits.

    Returns:
        torch.Tensor: Contrastive loss.
    """
    
    # Reshape to (batch_size * trajectory_length, emb_size)
    pred_flat = predictions.view(-1, predictions.size(-1))
    target_flat = targets.view(-1, targets.size(-1))
    
    pred_flat = pred_flat / pred_flat.norm(dim=1, keepdim=True)
    target_flat = target_flat / target_flat.norm(dim=1, keepdim=True)
    
    logits = torch.mm(pred_flat, target_flat.T) / temperature

    labels = torch.arange(pred_flat.size(0), device=predictions.device)
    
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss


def compute_loss(predictions, targets, reg_weight=0.1, contrast_weight=0.1):
    """
    Computes total loss including MSE, variance, covariance, and contrastive losses.

    Args:
        predictions (torch.Tensor): Predicted embeddings.
        targets (torch.Tensor): Target embeddings.
        reg_weight (float): Weight for regularization losses.
        contrast_weight (float): Weight for contrastive loss.

    Returns:
        torch.Tensor: Total loss.
    """

    mse_loss = nn.MSELoss()(predictions, targets)
    var_loss = variance_regularization(predictions)
    cov_loss = covariance_regularization(predictions)
    contrast_loss = contrastive_loss(predictions, targets)

    total_loss = mse_loss + reg_weight * (var_loss + cov_loss) + contrast_weight * contrast_loss
    
    return total_loss


def train_model(
    model, dataloader, optimizer, epochs, device, patience=5, save_path="pretrained_jepa_model.pth"
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

            # Compute loss
            loss = compute_loss(predictions, targets, reg_weight=0.2, contrast_weight=0.2)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {e + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
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

            # if patience_counter >= patience:
            #     print(
            #         f"Early stopping triggered after epoch {e+1}. Best epoch was {best_epoch+1}"
            #     )
            #     break


def main():
    save_path = "model_weights.pth"
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

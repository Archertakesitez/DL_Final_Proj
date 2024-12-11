import torch
import torch.nn as nn
import numpy as np
from dataset import WallDataset, create_wall_dataloader
from jepa_models import RecurrentJEPA, vicreg_loss
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


def train_step(model, optimizer, states, actions):
    # Forward pass
    predictions, targets = model(states, actions, training=True)

    # Calculate VICReg loss (imported from jepa_models)
    loss = vicreg_loss(predictions, targets)

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target encoder with momentum
    model._momentum_update_target_encoder()

    return loss.item()


def train_model(
    model,
    dataloader,
    optimizer,
    scheduler,
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
            loss = vicreg_loss(predictions, targets)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update target encoder with momentum
            model._momentum_update_target_encoder()

            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(
            f"Epoch {e + 1}/{epochs}, Training Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Update learning rate
        scheduler.step()

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

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Train the model
    train_model(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        epochs,
        device,
        patience=patience,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()

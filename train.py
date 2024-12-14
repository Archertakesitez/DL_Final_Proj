import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models import JEPA
from dataset import WallDataset, create_wall_dataloader


def train(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            states = batch.states.to(device)  # [B, T, C, H, W]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            optimizer.zero_grad()
            pred_encs, target_encs = model(states, actions)
            B, T_minus1, D = pred_encs.size()
            loss = model.compute_loss(
                pred_encs.view(B * T_minus1, D), target_encs.view(B * T_minus1, D)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/scratch/DL24FA/train"
    batch_size = 64
    epochs = 10
    repr_dim = 256
    learning_rate = 1e-3

    dataset = WallDataset(data_path=data_path, probing=False, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = JEPA(repr_dim=repr_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, epochs, device)

    # Save the trained model
    save_path = "model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

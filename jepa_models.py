import torch
from timm import create_model
import torch.nn as nn
from dataset import WallDataset
from torch.nn.functional import mse_loss
import torch.nn.functional as F


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ViTEncoder, self).__init__()
        self.vit = create_model(
            "vit_base_patch16_224",  # ViT base model with patch size 16
            pretrained=False,  # Do not load pretrained weights
            img_size=65,  # Input image size
            in_chans=2,  # Number of input channels
            num_classes=0,  # Remove classification head
        )
        self.projection = nn.Linear(embed_dim, embed_dim)  # Optional projection layer

    def forward(self, x):
        # x = x.flatten(1, 2)  # Flatten (2, 64, 64) to (128, 64)
        x = self.vit(x)
        return self.projection(x)


# class RecurrentPredictor(nn.Module):
#     def __init__(self, embed_dim=768, action_dim=2, hidden_dim=512):
#         super(RecurrentPredictor, self).__init__()
#         self.rnn = nn.GRU(embed_dim + action_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, embed_dim)

#     def forward(self, state_embeddings, actions):
#         # Concatenate state embeddings and actions along time dimension
#         combined = torch.cat([state_embeddings, actions], dim=-1)
#         rnn_out, _ = self.rnn(combined)  # RNN outputs hidden states over time
#         return self.fc(rnn_out)  # Map RNN hidden states to predicted embeddings


class RecurrentPredictor(nn.Module):
    def __init__(self, embed_dim=768, action_dim=2):
        super().__init__()
        # Wider network with residual connections
        self.fc1 = nn.Linear(embed_dim + action_dim, embed_dim * 4)
        self.ln1 = nn.LayerNorm(embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.ln2 = nn.LayerNorm(embed_dim * 2)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Residual connection
        self.residual = nn.Linear(embed_dim + action_dim, embed_dim)

    def forward(self, state_embedding, action):
        identity = state_embedding
        combined = torch.cat([state_embedding, action], dim=-1)

        # Main path
        x = self.ln1(torch.relu(self.fc1(combined)))
        x = self.dropout(x)
        x = self.ln2(torch.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.ln3(self.fc3(x))

        # Residual connection
        res = self.residual(combined)
        return x + res


# class RecurrentJEPA(nn.Module):
#     def __init__(self, embed_dim=768, action_dim=2, hidden_dim=512):
#         super(RecurrentJEPA, self).__init__()
#         self.encoder = ViTEncoder(embed_dim=embed_dim)
#         self.target_encoder = ViTEncoder(embed_dim=embed_dim)
#         self.predictor = RecurrentPredictor(embed_dim=embed_dim, action_dim=action_dim, hidden_dim=hidden_dim)

#     def forward(self, states, actions):
#         num_trajectories, trajectory_length, _, _, _ = states.shape
#         device = states.device

#         # Initialize lists to store embeddings
#         encoded_states = []
#         target_states = []

#         # Encode states timestep by timestep
#         for t in range(trajectory_length):
#             # Encode current timestep states
#             encoded_state = self.encoder(states[:, t])  # (num_trajectories, embed_dim)
#             target_state = self.target_encoder(states[:, t])  # Target encoder
#             encoded_states.append(encoded_state)
#             target_states.append(target_state)

#         # Stack encoded states across time
#         encoded_states = torch.stack(encoded_states, dim=1)  # (num_trajectories, trajectory_length, embed_dim)
#         target_states = torch.stack(target_states, dim=1)  # (num_trajectories, trajectory_length, embed_dim)

#         # Pass state-action sequence to predictor
#         predicted_embeddings = self.predictor(
#             encoded_states[:, :-1],  # Exclude the last state for predictions
#             actions
#         )

#         return predicted_embeddings, target_states[:, 1:]  # Exclude the initial state from targets


class RecurrentJEPA(nn.Module):
    def __init__(self, embed_dim=768, action_dim=2, momentum=0.99):
        super().__init__()
        self.encoder = ViTEncoder(embed_dim=embed_dim)
        self.target_encoder = ViTEncoder(embed_dim=embed_dim)
        self.predictor = RecurrentPredictor(embed_dim=embed_dim, action_dim=action_dim)
        self.repr_dim = embed_dim
        self.momentum = momentum

        # Initialize target encoder as a copy of the encoder
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(
                param_q.data
            )  # Initialize target encoder with encoder weights
            param_k.requires_grad = (
                False  # Target encoder should not be updated by gradients
            )

        # Consider adding dropout for regularization
        self.dropout = nn.Dropout(0.1)  # Optional

        # Initialize momentum parameters
        self.m = momentum
        self.register_buffer("m_factor", torch.tensor(1.0))  # For exponential schedule

    def forward(self, states, actions, training=True):
        batch_size, trajectory_length, _, _, _ = states.shape

        if training:
            # Get all target embeddings first
            target_embeddings = [
                self.target_encoder(states[:, t]) for t in range(trajectory_length)
            ]
            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)  # [B, T-1, embed_dim]
            targets = torch.stack(target_embeddings[1:], dim=1)  # [B, T-1, embed_dim]

            # Reshape for VICReg loss: combine batch and time dimensions
            predictions_flat = predictions.reshape(
                -1, predictions.shape[-1]
            )  # [B*(T-1), embed_dim]
            targets_flat = targets.reshape(
                -1, targets.shape[-1]
            )  # [B*(T-1), embed_dim]

            return predictions_flat, targets_flat
        else:
            # For evaluation/inference, return predictions in format [T, B, D]
            s_encoded = self.encoder(states[:, 0])
            predictions = []
            for t in range(actions.shape[1]):
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=0)  # [T, B, D]
            return predictions

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """Momentum update with exponential schedule"""
        # Convert to tensor and stay on same device
        self.m_factor.mul_(1.005).clamp_(max=self.m)  # In-place operations on tensor

        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.m_factor + param_q.data * (
                1.0 - self.m_factor
            )


def vicreg_loss(pred, target, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    # Invariance loss
    sim_loss = F.mse_loss(pred, target)

    # Variance loss
    std_pred = torch.sqrt(pred.var(dim=0) + 1e-4)
    std_target = torch.sqrt(target.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

    # Covariance loss
    pred_centered = pred - pred.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    cov_pred = (pred_centered.T @ pred_centered) / (pred.shape[0] - 1)
    cov_target = (target_centered.T @ target_centered) / (target.shape[0] - 1)
    cov_loss = off_diagonal(cov_pred).pow_(2).sum() / pred.shape[1]

    return sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss


def off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

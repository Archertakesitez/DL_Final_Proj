import torch
from timm import create_model
import torch.nn as nn
from dataset import WallDataset
from torch.nn.functional import mse_loss


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=768, in_channels=2):
        super(CNNEncoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 65, 65)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.size(0)

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flattened_size, embed_dim))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


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
        super(RecurrentPredictor, self).__init__()
        self.fc1 = nn.Linear(embed_dim + action_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, state_embedding, action):
        x = torch.cat(
            [state_embedding, action], dim=-1
        )  # Concatenate embeddings and action
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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
        super(RecurrentJEPA, self).__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.target_encoder = CNNEncoder(embed_dim=embed_dim)
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

    def forward(self, states, actions, training=True):
        batch_size = states.shape[0]

        if training:
            # Original JEPA training code
            target_embeddings = [
                self.target_encoder(states[:, t]) for t in range(states.shape[1])
            ]
            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)
            targets = torch.stack(target_embeddings[1:], dim=1)
            return predictions, targets
        else:
            # Evaluation mode - use only initial state
            assert states.shape[1] == 1, "Evaluation expects only initial state"
            s_encoded = self.encoder(states[:, 0])

            predictions = []
            for t in range(actions.shape[1]):
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)
            return predictions

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """Momentum update for target encoder"""
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

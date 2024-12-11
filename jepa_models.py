import torch
from timm import create_model
import torch.nn as nn
from dataset import WallDataset
from torch.nn.functional import mse_loss


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
    def __init__(self, embed_dim=768, action_dim=2):
        super(RecurrentJEPA, self).__init__()
        self.encoder = ViTEncoder(embed_dim=embed_dim)
        self.target_encoder = ViTEncoder(embed_dim=embed_dim)  # Optionally shared
        self.predictor = RecurrentPredictor(embed_dim=embed_dim, action_dim=action_dim)
        self.repr_dim = embed_dim  # Set repr_dim to the embedding dimension (768)

    def forward(self, states, actions, training=True):
        batch_size, trajectory_length, _, _, _ = states.shape

        if training:
            # Use all timesteps during training
            with torch.no_grad():
                target_embeddings = [
                    self.target_encoder(states[:, t]) for t in range(trajectory_length)
                ]
                targets = torch.stack(target_embeddings[1:], dim=1)
            
            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)

            return predictions, targets
        else:
            # Use only the first timestep during inference
            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                s_encoded = self.predictor(s_encoded, actions[:, t])
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)
            return predictions

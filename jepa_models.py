import torch
from timm import create_model
import torch.nn as nn
from dataset import WallDataset
from torch.nn.functional import mse_loss
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet18



class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ViTEncoder, self).__init__()
        self.vit = VisionTransformer(
            img_size=65,  # Input image size (65x65)
            patch_size=16,  # Patch size (16x16)
            in_chans=2,  # Number of input channels (e.g., 2-channel images)
            num_classes=0,  # Remove classification head for feature extraction
            embed_dim=embed_dim,  # Embedding dimension for tokens
            depth=24,  # Number of transformer layers
            num_heads=16,  # Number of attention heads
            mlp_ratio=4.0,  # MLP hidden layer size = embed_dim * 4
            qkv_bias=True,  # Allow biases in query/key/value projections
            norm_layer=nn.LayerNorm  # Use LayerNorm
        )
        self.projection = nn.Linear(embed_dim, embed_dim)  # Optional projection layer

    def forward(self, x):
        # x = x.flatten(1, 2)  # Flatten (2, 64, 64) to (128, 64)
        x = self.vit(x)
        return self.projection(x)
    

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=512, clip_value=1.0):
        """
        Custom ResNet model for 65x65 images.

        Args:
            embed_dim (int): Size of the final embedding dimension.
        """
        super(ResNetEncoder, self).__init__()

        self.clip_value = clip_value

        # Load ResNet-18 backbone
        self.resnet = resnet18(pretrained=False)
        
        # Modify the first convolution layer to handle 2-channel inputs
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove max-pooling to preserve more spatial information
        self.resnet.maxpool = nn.Identity()
        
        # Replace the fully connected layer with a projection layer
        self.resnet.fc = nn.Linear(512, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.2)

    
    def forward(self, x):
        """
        Forward pass through the custom ResNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, 65, 65).

        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, embed_dim).
        """

        out = self.resnet(x)

        return self.dropout(self.norm(x))
    

    def clip_gradients(self):
        """
        Clip gradients to stabilize training.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)


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
    def __init__(self, embed_dim=512, proj_dim=128):
        super(RecurrentPredictor, self).__init__()
        self.fc1 = nn.Linear(embed_dim + proj_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, state_embedding, action):
        x = torch.cat(
            [state_embedding, action], dim=-1
        )  # Concatenate embeddings and action

        x = self.norm(self.fc1(x))
        x = torch.relu(x)
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
    def __init__(self, embed_dim=512, action_dim=2, proj_dim=128):
        super(RecurrentJEPA, self).__init__()
        # self.encoder = ViTEncoder(embed_dim=embed_dim)
        # self.target_encoder = ViTEncoder(embed_dim=embed_dim)
        
        self.encoder = ResNetEncoder(embed_dim=embed_dim)
        self.target_encoder = ResNetEncoder(embed_dim=embed_dim)
        self.action_proj = nn.Linear(action_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.predictor = RecurrentPredictor(embed_dim=embed_dim, action_dim=action_dim)
        self.repr_dim = embed_dim
        # self.momentum = momentum

        # # Initialize target encoder as a copy of the encoder
        # for param_q, param_k in zip(
        #     self.encoder.parameters(), self.target_encoder.parameters()
        # ):
        #     param_k.data.copy_(
        #         param_q.data
        #     )  # Initialize target encoder with encoder weights
        #     param_k.requires_grad = (
        #         False  # Target encoder should not be updated by gradients
        #     )

        # Consider adding dropout for regularization
        self.dropout = nn.Dropout(0.2)  # Optional

    def forward(self, states, actions, training=True):
        batch_size, trajectory_length, _, _, _ = states.shape

        if training:
            # Use all timesteps during training
            target_embeddings = [
                self.target_encoder(states[:, t]) for t in range(trajectory_length)
            ]

            targets = torch.stack(target_embeddings[1:], dim=1)

            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                action_proj = self.action_proj(actions[:, t])
                s_encoded = self.predictor(s_encoded, action_proj)
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)

            return predictions, targets
        else:
            # Use only the first timestep during inference
            s_encoded = self.encoder(states[:, 0])  # Initial state embedding

            predictions = []
            for t in range(actions.shape[1]):  # trajectory_length - 1
                action_proj = self.proj_norm(self.action_proj(actions[:, t]))
                s_encoded = self.predictor(s_encoded, action_proj)
                predictions.append(s_encoded)

            predictions = torch.stack(predictions, dim=1)

            return predictions
        

    def clip_gradients(self):
        """
        Clip gradients for all submodules of RecurrentJEPA.
        """
        # Clip gradients for the encoder
        self.encoder.clip_gradients()

        # Clip gradients for the target encoder
        self.target_encoder.clip_gradients()

        # Optionally clip gradients for the predictor (if needed)
        for param in self.predictor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)


    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """Momentum update for target encoder"""
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

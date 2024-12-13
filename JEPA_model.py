import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


class CNNEncoder(nn.Module):
    """
    Implements Encθ(oₙ) from the JEPA formulation.
    Maps observations to latent representations.
    """

    def __init__(self, latent_dim=256):
        super().__init__()
        # Input: (B, 2, 65, 65) - 2 channels: agent position and wall/border layout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256, use_momentum=True, momentum=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.repr_dim = latent_dim  # Required by evaluator.py

        # Online networks
        self.encoder = ViTEncoder(latent_dim)
        self.action_encoder = nn.Linear(2, latent_dim)
        self.predictor = Predictor(latent_dim, action_dim=latent_dim)

        # Target network (momentum-updated)
        if use_momentum:
            self.target_encoder = ViTEncoder(latent_dim)
            # Initialize target network with same weights
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Update target network using momentum"""
        if not self.use_momentum:
            return

        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    def forward(self, states, actions):
        """
        Forward pass implementing recurrent JEPA prediction.
        Args:
            states: Observations [B, 1, C, H, W]  # Only initial state
            actions: Action sequence [B, T-1, 2]  # T-1 actions
        Returns:
            predictions: Predicted latent states [B, T, D]  # T total predictions
        """
        B = states.shape[0]
        T = actions.shape[1] + 1  # Total timesteps = num_actions + 1

        # Initial encoding
        curr_state = self.encoder(states[:, 0])  # [B, D]
        predictions = [curr_state]

        # Predict future states
        for t in range(T - 1):
            curr_action = actions[:, t]  # [B, 2]
            curr_state = self.predictor(curr_state, curr_action)
            predictions.append(curr_state)

        predictions = torch.stack(predictions, dim=1)  # [B, T, D]
        return predictions

    def compute_target(self, states):
        """Compute target representations for all states"""
        B, T = states.shape[:2]
        target_encoder = self.target_encoder if self.use_momentum else self.encoder

        targets = []
        for t in range(T):
            with torch.no_grad():
                target = target_encoder(states[:, t])
                targets.append(target)

        return torch.stack(targets, dim=1)  # [B, T, D]

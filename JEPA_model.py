import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Implements Encθ(oₙ) from the JEPA formulation.
    Maps observations to latent representations.
    """

    def __init__(self, latent_dim=256):
        super().__init__()
        # Input: (B, 2, 64, 64) - 2 channels: agent position and wall/border layout
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

        # Add self-attention layer for better wall/door understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        
        # Apply attention
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_att, _ = self.attention(x_flat, x_flat, x_flat)
        x_att = x_att.transpose(1, 2).reshape(B, C, H, W)
        
        # Combine attention output with original features
        x = x + x_att  # Residual connection
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, dim=-1)  # Normalize output representations


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        next_state = self.net(x)
        # Add residual connection to help maintain state information
        next_state = state + next_state
        return F.normalize(next_state, dim=-1)  # Normalize output


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256, use_momentum=True, momentum=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.repr_dim = latent_dim  # Required by evaluator.py

        # Online networks
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)

        # Target network (momentum-updated)
        if use_momentum:
            self.target_encoder = Encoder(latent_dim)
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

    def forward(self, initial_states, actions):
        """
        Forward pass implementing recurrent JEPA prediction.
        Args:
            initial_states: Initial observations [B, 2, H, W]  # Only initial state
            actions: Action sequence [B, T, 2]  # T actions for T+1 total timesteps
        Returns:
            predictions: Predicted latent states [B, T+1, D]  # T+1 total predictions including initial state
        """
        B = initial_states.shape[0]
        T = actions.shape[1]  # Number of actions

        # Initial encoding
        curr_state = self.encoder(initial_states)  # [B, D]
        predictions = [curr_state]

        # Predict future states
        for t in range(T):
            curr_action = actions[:, t]  # [B, 2]
            curr_state = self.predictor(curr_state, curr_action)
            predictions.append(curr_state)

        predictions = torch.stack(predictions, dim=1)  # [B, T+1, D]
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

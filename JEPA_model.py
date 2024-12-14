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

    def predict_future_state(self, state, action):
        """Helper to predict next state given current state and action"""
        return self.predictor(state, action)

    def forward(self, states, actions, teacher_forcing_ratio=0.5):
        """
        Forward pass implementing recurrent JEPA prediction with teacher forcing.
        Args:
            states: Observations [B, T, C, H, W]  # Full sequence for teacher forcing
            actions: Action sequence [B, T-1, 2]  # T-1 actions
            teacher_forcing_ratio: Probability of using teacher forcing (0.5 default)
        Returns:
            predictions: Predicted latent states [B, T, D]  # T total predictions
            targets: Target latent states [B, T, D]  # For training
        """
        B = states.shape[0]
        T = actions.shape[1] + 1  # Total timesteps = num_actions + 1

        # Initial encoding
        z_t = self.encoder(states[:, 0])  # [B, D]
        predictions = [z_t]

        # Get target encodings for teacher forcing
        with torch.no_grad():
            targets = self.compute_target(states)

        # Predict future states with teacher forcing
        for t in range(T - 1):
            # Teacher forcing: use ground truth previous state with probability
            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                z_t = self.predict_future_state(targets[:, t], actions[:, t])
            else:
                z_t = self.predict_future_state(z_t, actions[:, t])
            predictions.append(z_t)

        predictions = torch.stack(predictions, dim=1)  # [B, T, D]
        return predictions, targets

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

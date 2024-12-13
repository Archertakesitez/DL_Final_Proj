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
    def __init__(self, latent_dim=256, momentum=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.momentum = momentum
        self.repr_dim = latent_dim

        # Online networks for state prediction
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)

        # Target network with momentum update (LeCun's approach)
        self.target_encoder = Encoder(latent_dim)
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """LeCun's momentum update for target network"""
        for param_q, param_k in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    def predict_future_state(self, current_state, action):
        """LeCun's approach: predict next state given current state and action"""
        return self.predictor(current_state, action)

    def compute_target(self, states):
        """Get target representations using momentum encoder"""
        targets = []
        with torch.no_grad():  # Stop gradient for targets (LeCun's approach)
            for t in range(states.shape[1]):
                target = self.target_encoder(states[:, t])
                targets.append(target)
        return torch.stack(targets, dim=1)

    def forward(self, states, actions):
        """
        LeCun's JEPA forward pass: predict future states in latent space
        Args:
            states: [B, T, 2, 65, 65] - Sequence of states
            actions: [B, T-1, 2] - Sequence of actions
        Returns:
            predictions: [B, T, D] - Predicted state representations
        """
        # Initial state encoding
        z_t = self.encoder(states[:, 0])
        predictions = [z_t]

        # Predict future states autoregressively (LeCun's approach)
        for t in range(actions.shape[1]):
            z_t = self.predict_future_state(z_t, actions[:, t])
            predictions.append(z_t)

        return torch.stack(predictions, dim=1)

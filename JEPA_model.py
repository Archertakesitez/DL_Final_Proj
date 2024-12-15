import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block for preserving spatial information
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Implements Encθ(oₙ) from the JEPA formulation.
    Maps observations to latent representations.
    """

    def __init__(self, latent_dim=256):
        super().__init__()
        # Input: (B, 2, 65, 65) - 2 channels: agent position and wall/border layout

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # -> (32, 65, 65)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        # Only two residual blocks at critical points
        self.res1 = ResidualBlock(
            32, 64
        )  # First residual to preserve initial spatial info
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (64, 32, 32)

        # Regular convolutions for middle layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> (128, 16, 16)
        )

        # Second residual before final spatial reduction
        self.res2 = ResidualBlock(
            128, 256
        )  # Second residual to preserve important features
        self.pool2 = nn.MaxPool2d(2, 2)  # -> (256, 8, 8)

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(True), nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)

        # First residual + pooling
        x = self.pool1(self.res1(x))

        # Middle convolution
        x = self.conv1(x)

        # Second residual + pooling
        x = self.pool2(self.res2(x))

        # FC layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(True),
            nn.Linear(384, latent_dim),
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

    def forward(self, states, actions):
        """
        Forward pass implementing recurrent JEPA prediction.
        Args:
            states: Initial observation only [B, 1, C, H, W]
            actions: Full action sequence [B, T, 2]
        Returns:
            predictions: Predicted latent states [B, T, D]
            targets: Target latent states [B, T, D]
        """
        B, T = actions.shape[:2]  # Get batch size and sequence length from actions

        # Initial encoding (Enc_θ)
        s0 = self.encoder(states[:, 0])  # [B, D]
        t0 = self.target_encoder(states[:, 0]) if self.use_momentum else s0
        # Predict future states recursively (Pred_φ)
        predictions = [s0]
        targets = [t0]

        for t in range(T):  # T predictions
            # Use previous prediction and current action to predict next state
            pred_t = self.predictor(predictions[-1], actions[:, t])
            targ_t = self.predictor(
                targets[-1], actions[:, t]
            )  # Use same predictor for target
            predictions.append(pred_t)
            targets.append(targ_t)

        predictions = torch.stack(predictions, dim=1)  # [B, T+1, D]
        targets = torch.stack(targets, dim=1)  # [B, T+1, D]

        return predictions, targets

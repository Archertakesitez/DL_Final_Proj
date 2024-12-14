from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class Encoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(
                2, 32, kernel_size=3, stride=2, padding=1
            ),  # [BS, 2, 64, 64] -> [BS, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [BS, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [BS, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [BS, 256, 4, 4]
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, repr_dim)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim, action_dim=2):
        super().__init__()
        self.fc = build_mlp([repr_dim + action_dim, 512, repr_dim])

    def forward(self, s_prev, u_prev):
        x = torch.cat([s_prev, u_prev], dim=-1)
        x = self.fc(x)
        return x


class TargetEncoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.encoder = Encoder(repr_dim)
        for param in self.parameters():
            param.requires_grad = False  # Freeze target encoder

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)


class JEPA(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = Encoder(repr_dim)
        self.predictor = Predictor(repr_dim)
        self.target_encoder = TargetEncoder(repr_dim)

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, C, H, W]
            actions: [B, T-1, 2]
        Returns:
            pred_encs: [B, T-1, repr_dim]
        """
        B, T, C, H, W = states.size()
        enc_states = self.encoder(states.view(B * T, C, H, W))
        enc_states = enc_states.view(B, T, -1)

        s_prev = enc_states[:, 0]
        pred_encs = []
        for t in range(T - 1):
            u_prev = actions[:, t]
            s_pred = self.predictor(s_prev, u_prev)
            pred_encs.append(s_pred)
            s_prev = s_pred

        pred_encs = torch.stack(pred_encs, dim=1)  # [B, T-1, repr_dim]

        # Target representations
        with torch.no_grad():
            target_encs = self.target_encoder(states[:, 1:])  # [B, T-1, repr_dim]

        return pred_encs, target_encs

    def compute_loss(self, pred_encs, target_encs):
        """
        Compute the JEPA loss with variance and covariance regularization to prevent collapse.
        """
        repr_loss = F.mse_loss(pred_encs, target_encs)

        # Variance regularization
        std_pred = torch.sqrt(pred_encs.var(dim=0) + 1e-4)
        std_target = torch.sqrt(target_encs.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # Covariance regularization
        pred_encs_centered = pred_encs - pred_encs.mean(dim=0)
        target_encs_centered = target_encs - target_encs.mean(dim=0)

        cov_pred = (pred_encs_centered.T @ pred_encs_centered) / (
            pred_encs_centered.size(0) - 1
        )
        cov_target = (target_encs_centered.T @ target_encs_centered) / (
            target_encs_centered.size(0) - 1
        )

        cov_loss_pred = (cov_pred - torch.diag(torch.diag(cov_pred))).pow(
            2
        ).sum() / self.repr_dim
        cov_loss_target = (cov_target - torch.diag(torch.diag(cov_target))).pow(
            2
        ).sum() / self.repr_dim

        cov_loss = cov_loss_pred + cov_loss_target

        loss = repr_loss + 25 * std_loss + 1 * cov_loss
        return loss

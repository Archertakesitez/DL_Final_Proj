import torch
from timm import create_model
import torch.nn as nn


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ViTEncoder, self).__init__()
        self.vit = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=64,  # Corrected to match input size
            in_chans=2,   # 2 channels: agent and walls
            num_classes=0,
            patch_size=8  # Smaller patch size for 64x64 images
        )
        # Simple projection to ensure stable training
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x = self.vit(x)
        return self.projection(x)


class RecurrentPredictor(nn.Module):
    def __init__(self, embed_dim=768, action_dim=2):
        super(RecurrentPredictor, self).__init__()
        # Simple MLP for state-action prediction
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, state_embedding, action):
        """
        Implements s̃ₙ = Predφ(s̃ₙ₋₁, uₙ₋₁)
        """
        x = torch.cat([state_embedding, action], dim=-1)
        return self.net(x)


class RecurrentJEPA(nn.Module):
    def __init__(self, embed_dim=768, action_dim=2):
        super(RecurrentJEPA, self).__init__()
        self.encoder = ViTEncoder(embed_dim=embed_dim)
        self.target_encoder = ViTEncoder(embed_dim=embed_dim)
        self.predictor = RecurrentPredictor(embed_dim=embed_dim, action_dim=action_dim)
        self.repr_dim = embed_dim
        
        # Initialize target encoder as copy of encoder
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_target_encoder(self, momentum=0.99):
        """
        Momentum update of target encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def forward(self, states, actions, training=True):
        """
        Implements recurrent JEPA:
        s̃₀ = s₀ = Encθ(o₀)
        s̃ₙ = Predφ(s̃ₙ₋₁, uₙ₋₁)
        """
        batch_size, trajectory_length, _, _, _ = states.shape
        
        if training:
            # Update target encoder
            self._momentum_update_target_encoder()
            
            # Initial state encoding: s̃₀ = s₀ = Encθ(o₀)
            s_tilde = self.encoder(states[:, 0])
            
            # Get target encodings
            with torch.no_grad():
                target_states = [
                    self.target_encoder(states[:, t]) 
                    for t in range(1, trajectory_length)
                ]
                targets = torch.stack(target_states, dim=1)
            
            # Recurrent predictions: s̃ₙ = Predφ(s̃ₙ₋₁, uₙ₋₁)
            predictions = []
            for t in range(trajectory_length - 1):
                s_tilde = self.predictor(s_tilde, actions[:, t])
                predictions.append(s_tilde)
            
            predictions = torch.stack(predictions, dim=1)
            return predictions, targets
        else:
            # For inference/probing
            s_tilde = self.encoder(states[:, 0])
            predictions = []
            for t in range(actions.shape[1]):
                s_tilde = self.predictor(s_tilde, actions[:, t])
                predictions.append(s_tilde)
            return torch.stack(predictions, dim=1)

from __future__ import annotations

import torch
from torch import nn


class RoutingMLP(nn.Module):
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64, num_routes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.route_classifier = nn.Linear(hidden_dim, num_routes)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "route_logits": self.route_classifier(hidden),
            "confidence_logit": self.confidence_head(hidden).squeeze(-1),
        }

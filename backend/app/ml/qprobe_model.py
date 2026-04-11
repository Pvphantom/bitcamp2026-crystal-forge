"""ML-QProbe recommender model."""

from __future__ import annotations

import torch
from torch import nn


class QProbeMLP(nn.Module):
    def __init__(self, *, input_dim: int = 15, hidden_dim: int = 64, num_cost_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cost_head = nn.Linear(hidden_dim, num_cost_classes)
        self.success_head = nn.Linear(hidden_dim, 2)
        self.error_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "cost_logits": self.cost_head(hidden),
            "success_logits": self.success_head(hidden),
            "error_pred": self.error_head(hidden).squeeze(-1),
        }

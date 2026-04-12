from __future__ import annotations

import torch
from torch import nn


class AdaptiveStopMLP(nn.Module):
    def __init__(self, *, input_dim: int = 41, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.margin_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "stop_logit": self.stop_head(hidden).squeeze(-1),
            "margin_pred": self.margin_head(hidden).squeeze(-1),
        }

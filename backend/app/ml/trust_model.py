from __future__ import annotations

import torch
from torch import nn


class TrustMLP(nn.Module):
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.error_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "risk_logits": self.classifier(hidden),
            "error_pred": self.error_head(hidden).squeeze(-1),
        }

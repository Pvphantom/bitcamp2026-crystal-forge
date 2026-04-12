from __future__ import annotations

import torch
from torch import nn


class StructuredCorrMapMLP(nn.Module):
    def __init__(self, input_dim: int = 49, hidden_dim: int = 96, num_intrinsic_labels: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_field_head = nn.Linear(hidden_dim, 1)
        self.quantum_frontier_head = nn.Linear(hidden_dim, 1)
        self.intrinsic_classifier = nn.Linear(hidden_dim, num_intrinsic_labels)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "mean_field_logit": self.mean_field_head(hidden).squeeze(-1),
            "quantum_frontier_logit": self.quantum_frontier_head(hidden).squeeze(-1),
            "intrinsic_logits": self.intrinsic_classifier(hidden),
        }

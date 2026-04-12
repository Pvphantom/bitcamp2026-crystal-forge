from __future__ import annotations

import torch
from torch import nn


class SuperconductorQProbeMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.safe_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.safe_head(self.net(features)).squeeze(-1)

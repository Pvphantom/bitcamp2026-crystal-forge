"""Graph-aware phase classifier implemented in plain PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        # nodes: [B, N, H]
        neighbor_states = nodes.unsqueeze(1).expand(-1, nodes.size(1), -1, -1)
        gate = self.edge_gate(edge_attr) * adjacency.unsqueeze(-1)
        messages = (gate * neighbor_states).sum(dim=2)
        degree = adjacency.sum(dim=2, keepdim=True).clamp_min(1.0)
        messages = messages / degree
        updated = self.update(torch.cat([nodes, messages], dim=-1))
        masked = updated * node_mask.unsqueeze(-1)
        return self.norm(nodes + masked)


class HubbardPhaseGraphNet(nn.Module):
    def __init__(
        self,
        *,
        node_in: int = 5,
        edge_in: int = 1,
        global_in: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.global_proj = nn.Linear(global_in, 8)
        self.node_proj = nn.Linear(node_in + 8, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim, edge_in) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self,
        *,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        global_embedding = self.global_proj(global_feats)
        broadcast_global = global_embedding.unsqueeze(1).expand(-1, nodes.size(1), -1)
        hidden = self.node_proj(torch.cat([nodes, broadcast_global], dim=-1))
        hidden = hidden * node_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, edge_attr, node_mask)

        pooled = (hidden * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(pooled)

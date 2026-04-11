"""Dataset schema and graph-sample builders for Crystal Forge ML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


PHASE_LABELS = ["Metal", "Mott Insulator", "Antiferromagnet", "Singlet-rich"]
PHASE_TO_INDEX = {label: index for index, label in enumerate(PHASE_LABELS)}

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_TRAIN_DATASET = ARTIFACTS_DIR / "graphs_train.pt"
DEFAULT_TEST_DATASET = ARTIFACTS_DIR / "graphs_test.pt"
DEFAULT_2X2_BASE_DATASET = ARTIFACTS_DIR / "graphs_2x2_base.pt"
DEFAULT_2X3_BASE_DATASET = ARTIFACTS_DIR / "graphs_2x3_base.pt"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "phase_gnn.pt"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "gnn_metrics.json"


@dataclass
class GraphSample:
    nodes: torch.Tensor
    adjacency: torch.Tensor
    edge_attr: torch.Tensor
    node_mask: torch.Tensor
    global_feats: torch.Tensor
    label: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "adjacency": self.adjacency,
            "edge_attr": self.edge_attr,
            "node_mask": self.node_mask,
            "global_feats": self.global_feats,
            "label": self.label,
            "metadata": self.metadata,
        }


def classify_phase_rule(U: float, filling: float, ms2: float) -> str:
    """Operational training label rule for the current small-lattice dataset.

    On a 2x2 lattice, strong-coupling half-filled states almost always carry
    sizable antiferromagnetic correlations, so the original spec's strict
    Mott-vs-AFM split by ``Ms2`` alone collapses. For training, we keep the
    categories separable by using:

    - Metal: weak coupling
    - Mott Insulator: very strong coupling, near half filling, suppressed double
      occupancy (approximated here by the highest-U bucket)
    - Antiferromagnet: moderate-to-strong coupling, near half filling, large Ms2
    - Singlet-rich: everything else
    """
    if U < 2.0:
        return "Metal"
    if U >= 8.0 and abs(filling - 1.0) < 0.05:
        return "Mott Insulator"
    if U >= 4.0 and abs(filling - 1.0) < 0.05 and ms2 > 0.4:
        return "Antiferromagnet"
    return "Singlet-rich"


def build_graph_sample(
    *,
    Lx: int,
    Ly: int,
    site_features: list[list[float]],
    bond_strengths: dict[tuple[int, int], float],
    global_feats: list[float],
    label: str,
    metadata: dict[str, Any],
    max_nodes: int,
) -> GraphSample:
    Ns = Lx * Ly
    nodes = torch.zeros((max_nodes, len(site_features[0])), dtype=torch.float32)
    adjacency = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
    edge_attr = torch.zeros((max_nodes, max_nodes, 1), dtype=torch.float32)
    node_mask = torch.zeros(max_nodes, dtype=torch.float32)

    for i, features in enumerate(site_features):
        nodes[i] = torch.tensor(features, dtype=torch.float32)
        node_mask[i] = 1.0

    for (i, j), strength in bond_strengths.items():
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
        edge_attr[i, j, 0] = float(strength)
        edge_attr[j, i, 0] = float(strength)

    return GraphSample(
        nodes=nodes,
        adjacency=adjacency,
        edge_attr=edge_attr,
        node_mask=node_mask,
        global_feats=torch.tensor(global_feats, dtype=torch.float32),
        label=PHASE_TO_INDEX[label],
        metadata=metadata,
    )


def collate_graph_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[dict[str, Any]]]:
    return {
        "nodes": torch.stack([sample["nodes"] for sample in samples], dim=0),
        "adjacency": torch.stack([sample["adjacency"] for sample in samples], dim=0),
        "edge_attr": torch.stack([sample["edge_attr"] for sample in samples], dim=0),
        "node_mask": torch.stack([sample["node_mask"] for sample in samples], dim=0),
        "global_feats": torch.stack([sample["global_feats"] for sample in samples], dim=0),
        "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "metadata": [sample["metadata"] for sample in samples],
    }

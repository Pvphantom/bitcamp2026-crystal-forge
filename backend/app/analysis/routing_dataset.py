from __future__ import annotations

from typing import Any

import torch

from app.ml.schema import REFERENCE_QUALITY_LABELS, ROUTING_LABELS, RoutingBenchmarkSample


def validate_routing_sample(sample: RoutingBenchmarkSample) -> None:
    if sample.route_label not in ROUTING_LABELS:
        raise ValueError(f"Unsupported routing label: {sample.route_label}")
    if sample.reference_quality not in REFERENCE_QUALITY_LABELS:
        raise ValueError(f"Unsupported reference quality: {sample.reference_quality}")
    if sample.features.ndim != 1:
        raise ValueError("Routing benchmark features must be a 1D tensor")
    if not sample.feature_groups:
        raise ValueError("Routing benchmark sample must include at least one feature group")
    total_group_dim = 0
    for name, tensor in sample.feature_groups.items():
        if tensor.ndim != 1:
            raise ValueError(f"Feature group '{name}' must be a 1D tensor")
        total_group_dim += int(tensor.shape[0])
    if total_group_dim != int(sample.features.shape[0]):
        raise ValueError(
            f"Feature group dimensions ({total_group_dim}) do not match flattened feature dimension ({sample.features.shape[0]})"
        )
    if sample.reference_solver not in sample.solver_outcomes:
        raise ValueError("Reference solver must be present in solver outcomes")


def benchmark_sample_to_dict(sample: RoutingBenchmarkSample) -> dict[str, Any]:
    validate_routing_sample(sample)
    return sample.to_dict()


def flatten_feature_groups(feature_groups: dict[str, torch.Tensor], ordered_names: list[str] | None = None) -> torch.Tensor:
    names = ordered_names or list(feature_groups.keys())
    return torch.cat([feature_groups[name] for name in names], dim=0)

from __future__ import annotations

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_adaptive_features import build_qprobe_adaptive_feature_vector
from app.optimization.measurement_plan import AdaptiveMeasurementStep


def build_qprobe_adaptive_step_feature_vector(
    *,
    problem: ProblemSpec,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
    step: AdaptiveMeasurementStep,
    full_cost: int,
) -> torch.Tensor:
    base = build_qprobe_adaptive_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=target_names,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        readout_flip_prob=readout_flip_prob,
    )
    total_targets = max(1, len(target_names))
    current_cost = step.plan.cost
    step_features = torch.tensor(
        [
            float(step.step_index),
            float(current_cost),
            float(current_cost / max(1, full_cost)),
            float(len(step.covered_targets) / total_targets),
            float(len(step.unresolved_targets) / total_targets),
            float(step.max_uncertainty),
            float(step.max_uncertainty + readout_flip_prob),
            float(step.chosen_group.num_terms),
            * _basis_family_one_hot(step.chosen_group.basis),
        ],
        dtype=torch.float32,
    )
    return torch.cat([base, step_features], dim=0)


def qprobe_adaptive_step_feature_dim() -> int:
    return 44


def _basis_family_one_hot(basis: str) -> list[float]:
    active = {symbol for symbol in basis if symbol != "I"}
    if not active or active == {"Z"}:
        return [1.0, 0.0, 0.0, 0.0]
    if active == {"X"}:
        return [0.0, 1.0, 0.0, 0.0]
    if active == {"Y"}:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]

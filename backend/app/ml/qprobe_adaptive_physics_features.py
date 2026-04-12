from __future__ import annotations

import torch

from app.analysis.qprobe_physics_scorecard import build_qprobe_physics_scorecard
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_adaptive_step_features import build_qprobe_adaptive_step_feature_vector
from app.optimization.measurement_plan import AdaptiveMeasurementStep


def build_qprobe_adaptive_physics_feature_vector(
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
    base = build_qprobe_adaptive_step_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=target_names,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        readout_flip_prob=readout_flip_prob,
        step=step,
        full_cost=full_cost,
    )
    scorecard = build_qprobe_physics_scorecard(
        operator_map=operator_map,
        target_names=target_names,
        step=step,
        readout_flip_prob=readout_flip_prob,
    )
    extra = torch.tensor(
        [
            scorecard["compatibility_score"],
            scorecard["uncovered_weight_fraction"],
            scorecard["unresolved_mass_fraction"],
            scorecard["off_diagonal_burden"],
            scorecard["coefficient_concentration"],
            scorecard["target_overlap"],
            scorecard["uncertainty_pressure"],
            scorecard["chosen_basis_is_mixed"],
        ],
        dtype=torch.float32,
    )
    return torch.cat([base, extra], dim=0)


def qprobe_adaptive_physics_feature_dim() -> int:
    return 52

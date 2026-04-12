from __future__ import annotations

import torch

from app.analysis.qprobe_superconductor_channelize import build_superconductor_channel_plan
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_operator_features import build_qprobe_operator_feature_vector


SUPERCONDUCTOR_TARGETS = ("n", "D", "Ms2", "Cs_max", "K", "Pair_nn", "Pair_span")


def build_qprobe_superconductor_feature_vector(
    *,
    problem: ProblemSpec,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
) -> torch.Tensor:
    base = build_qprobe_operator_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=target_names,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        readout_flip_prob=readout_flip_prob,
    )
    plan = build_superconductor_channel_plan(operator_map=operator_map, targets=target_names)
    target_flags = [1.0 if name in target_names else 0.0 for name in SUPERCONDUCTOR_TARGETS]
    extras = torch.tensor(
        [
            float(plan.basis_concentration),
            float(plan.mean_support),
            float(plan.coherence_score),
            float(len(plan.channels)),
            float(sum(1 for name in target_names if "Pair" in name)),
            float(sum(1 for name in target_names if name in {"n", "D"})),
            float(sum(1 for name in target_names if name in {"Ms2", "Cs_max"})),
            float(sum(1 for name in target_names if name == "K")),
            *target_flags,
        ],
        dtype=torch.float32,
    )
    return torch.cat([base, extras], dim=0)


def qprobe_superconductor_feature_dim() -> int:
    return 28 + 8 + len(SUPERCONDUCTOR_TARGETS)

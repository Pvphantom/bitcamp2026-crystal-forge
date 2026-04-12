from __future__ import annotations

import torch

from app.analysis.synthetic_operator_families_generalized import (
    GENERALIZED_OPERATOR_FAMILIES,
    HARD_OPERATOR_FAMILIES,
    build_generalized_synthetic_operator_bundle,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_adaptive_step_features import qprobe_adaptive_step_feature_dim


def test_generalized_operator_families_include_harder_generic_cases() -> None:
    assert "diffuse_mixed" in GENERALIZED_OPERATOR_FAMILIES
    assert "basis_conflict" in GENERALIZED_OPERATOR_FAMILIES
    assert HARD_OPERATOR_FAMILIES <= set(GENERALIZED_OPERATOR_FAMILIES)


def test_generalized_bundle_builds_valid_operator_map() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=1.0, mu=0.0)
    bundle = build_generalized_synthetic_operator_bundle(
        problem=problem,
        family="diffuse_mixed",
        num_targets=3,
        seed=7,
    )
    assert len(bundle.operator_map) == 3
    assert bundle.target_names == tuple(bundle.operator_map.keys())
    for operator in bundle.operator_map.values():
        assert operator.num_qubits == problem.nqubits


def test_generalized_samples_can_match_existing_feature_dim() -> None:
    zeros = torch.zeros(qprobe_adaptive_step_feature_dim())
    assert zeros.shape[0] == qprobe_adaptive_step_feature_dim()

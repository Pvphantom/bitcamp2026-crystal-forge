from __future__ import annotations

import torch

from app.analysis.synthetic_operator_families import (
    SYNTHETIC_OPERATOR_FAMILIES,
    build_synthetic_operator_bundle,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_operator_features import build_qprobe_operator_feature_vector, qprobe_operator_feature_dim
from scripts.train_qprobe_adaptive_general_model import split_by_operator_family


def test_synthetic_operator_bundle_builds_supported_families() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=1.0, g=0.0)
    for family in SYNTHETIC_OPERATOR_FAMILIES:
        bundle = build_synthetic_operator_bundle(problem=problem, family=family, num_targets=2, seed=7)
        assert bundle.family == family
        assert len(bundle.operator_map) == 2
        assert bundle.target_names == tuple(bundle.operator_map.keys())


def test_synthetic_operator_features_match_general_dimension() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    bundle = build_synthetic_operator_bundle(problem=problem, family="mixed_local", num_targets=3, seed=11)
    features = build_qprobe_operator_feature_vector(
        problem=problem,
        operator_map=bundle.operator_map,
        target_names=bundle.target_names,
        tolerance=0.03,
        shots_per_group=2000,
        readout_flip_prob=0.02,
    )
    assert features.shape == torch.Size([qprobe_operator_feature_dim()])


def test_split_by_operator_family_prevents_family_leakage() -> None:
    samples = []
    for family in ("hubbard", "tfim"):
        for op_family in SYNTHETIC_OPERATOR_FAMILIES[:4]:
            for _ in range(3):
                samples.append(
                    {
                        "features": torch.zeros(qprobe_operator_feature_dim()),
                        "recommended_cost": 1,
                        "success": True,
                        "max_abs_error": 0.01,
                        "metadata": {"family": family, "operator_family": op_family},
                    }
                )
    train, val, test = split_by_operator_family(samples)
    train_groups = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in train}
    val_groups = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in val}
    test_groups = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in test}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

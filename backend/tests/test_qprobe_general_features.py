from __future__ import annotations

from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_operator_features import (
    build_qprobe_operator_feature_vector,
    qprobe_operator_feature_dim,
)
from app.observables.registry import build_default_observable_registry


def test_qprobe_general_feature_vector_has_expected_dimension() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    operator_map = registry.operator_map(problem)
    features = build_qprobe_operator_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=("D", "K"),
        tolerance=0.03,
        shots_per_group=4000,
        readout_flip_prob=0.02,
    )
    assert features.shape == (qprobe_operator_feature_dim(),)


def test_qprobe_general_feature_vector_works_for_tfim() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    operator_map = registry.operator_map(problem)
    features = build_qprobe_operator_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=("Mz", "ZZ_nn"),
        tolerance=0.03,
        shots_per_group=4000,
        readout_flip_prob=0.02,
    )
    assert features.shape == (qprobe_operator_feature_dim(),)
    assert float(features[1].item()) == 1.0

import torch

from app.analysis.solver_compare import compare_solver_results
from app.analysis.routing_dataset import benchmark_sample_to_dict
from app.analysis.trust_features import (
    TRUST_FEATURE_GROUPS,
    build_trust_feature_groups,
    build_trust_feature_vector,
    flatten_trust_feature_groups,
    trust_feature_group_dims,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import RoutingBenchmarkSample, SolverBenchmarkOutcome
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


def test_trust_feature_vector_shape_is_stable() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    cheap = MeanFieldSolver().solve(problem)
    features = build_trust_feature_vector(problem, cheap)
    assert features.shape == (22,)


def test_trust_feature_groups_are_stable_and_flatten_cleanly() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    cheap = MeanFieldSolver().solve(problem)
    feature_groups = build_trust_feature_groups(problem, cheap)
    expected_dims = trust_feature_group_dims()

    assert tuple(feature_groups.keys()) == TRUST_FEATURE_GROUPS
    assert {name: int(values.shape[0]) for name, values in feature_groups.items()} == expected_dims
    assert torch.equal(flatten_trust_feature_groups(feature_groups), build_trust_feature_vector(problem, cheap))
    assert flatten_trust_feature_groups(feature_groups, exclude_groups=("stability",)).shape == (14,)


def test_trust_oracle_labels_are_well_formed() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=8.0, mu=4.0)
    exact = ExactEDSolver().solve(problem)
    approx = MeanFieldSolver().solve(problem)
    comparison = compare_solver_results(problem, exact, approx)
    assert comparison.risk_label in {"safe", "warning", "unsafe"}
    assert comparison.max_abs_error >= 0.0
    assert comparison.energy_error >= 0.0


def test_tfim_trust_features_and_labels_are_well_formed() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    exact = ExactEDSolver().solve(problem)
    approx = TFIMMeanFieldSolver().solve(problem)
    features = build_trust_feature_vector(problem, approx)
    comparison = compare_solver_results(problem, exact, approx)
    assert features.shape == (22,)
    assert comparison.risk_label in {"safe", "warning", "unsafe"}
    assert comparison.max_abs_error >= 0.0


def test_routing_benchmark_sample_tracks_feature_groups_and_reference_provenance() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    cheap = MeanFieldSolver().solve(problem)
    groups = build_trust_feature_groups(problem, cheap)
    sample = RoutingBenchmarkSample(
        features=flatten_trust_feature_groups(groups),
        feature_groups=groups,
        route_label="mean_field",
        problem_metadata={"family": "hubbard", "Lx": 2, "Ly": 2},
        solver_outcomes={
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="classical",
                succeeded=True,
                runtime_s=0.01,
                observables={"D": float(cheap.global_observables["D"])},
            ),
        },
        reference_solver="mean_field",
        reference_quality="weak",
        label_source="cheap-consensus",
    )

    payload = benchmark_sample_to_dict(sample)
    assert payload["route_label"] == "mean_field"
    assert payload["reference_quality"] == "weak"
    assert tuple(payload["feature_groups"].keys()) == TRUST_FEATURE_GROUPS

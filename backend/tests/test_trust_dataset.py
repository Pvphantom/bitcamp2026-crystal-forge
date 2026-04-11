import torch

from app.analysis.solver_compare import compare_solver_results
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


def test_trust_feature_vector_shape_is_stable() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    cheap = MeanFieldSolver().solve(problem)
    features = build_trust_feature_vector(problem, cheap)
    assert features.shape == (22,)


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

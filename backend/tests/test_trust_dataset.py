import torch

from app.analysis.solver_compare import compare_solver_results
from app.domain.problem_spec import ProblemSpec
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from scripts.data_gen_trust import build_trust_feature_vector


def test_trust_feature_vector_shape_is_stable() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    features = build_trust_feature_vector(
        problem=problem,
        mean_field_globals={"D": 0.2, "n": 1.0, "Ms2": 0.5, "K": -0.7, "Cs_max": -0.1, "energy": -3.0},
        mean_field_sites={
            "n_up": [0.6, 0.4, 0.4, 0.6],
            "n_dn": [0.4, 0.6, 0.6, 0.4],
            "D_site": [0.24, 0.24, 0.24, 0.24],
            "Sz_site": [0.2, -0.2, -0.2, 0.2],
        },
        iterations=12,
        converged=True,
    )
    assert features.shape == (17,)


def test_trust_oracle_labels_are_well_formed() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=8.0, mu=4.0)
    exact = ExactEDSolver().solve(problem)
    approx = MeanFieldSolver().solve(problem)
    comparison = compare_solver_results(problem, exact, approx)
    assert comparison.risk_label in {"safe", "warning", "unsafe"}
    assert comparison.max_abs_error >= 0.0
    assert comparison.energy_error >= 0.0

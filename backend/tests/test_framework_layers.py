from app.analysis.solver_compare import compare_solver_results
from app.domain.problem_spec import ProblemSpec
from app.observables.registry import build_default_observable_registry
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.registry import SolverRegistry


def test_problem_spec_hubbard_properties() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=3, t=1.0, U=4.0, mu=2.0)
    assert problem.model_family == "hubbard"
    assert problem.nsites == 6
    assert problem.nqubits == 12
    assert problem.t == 1.0
    assert problem.U == 4.0
    assert problem.mu == 2.0


def test_solver_registry_wraps_exact_ed_solver() -> None:
    registry = SolverRegistry()
    exact = ExactEDSolver()
    registry.register(exact)
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    assert registry.supports("exact_ed", problem) is True
    result = registry.get("exact_ed").solve(problem)
    assert result.solver_name == "exact_ed"
    assert "D" in result.global_observables
    assert len(result.site_observables["n_up"]) == 4


def test_observable_registry_builds_hubbard_operator_map() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    operator_map = registry.operator_map(problem)
    assert set(operator_map) == {"D", "n", "Ms2", "K", "Cs_max"}
    bond_ops = registry.hubbard_bond_operators(problem)
    assert len(bond_ops) == 4


def test_mean_field_solver_returns_bounded_observables() -> None:
    solver = MeanFieldSolver()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    result = solver.solve(problem)
    assert result.solver_name == "mean_field"
    assert 0.0 <= result.global_observables["D"] <= 1.0
    assert 0.0 <= result.global_observables["n"] <= 2.0
    assert result.metadata["iterations"] >= 1
    assert len(result.site_observables["n_up"]) == 4
    assert len(result.bond_observables) == 4


def test_exact_vs_mean_field_comparison_produces_risk_label() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=8.0, mu=4.0)
    exact = ExactEDSolver().solve(problem)
    mean_field = MeanFieldSolver().solve(problem)
    comparison = compare_solver_results(problem, exact, mean_field)
    assert comparison.reference_solver == "exact_ed"
    assert comparison.candidate_solver == "mean_field"
    assert comparison.risk_label in {"safe", "warning", "unsafe"}
    assert comparison.max_abs_error >= 0.0
    assert comparison.energy_error >= 0.0

from app.domain.problem_spec import ProblemSpec
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.vqe import VQESettings, VQESolver


def test_tfim_vqe_solver_returns_expected_fields() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    solver = VQESolver(VQESettings(depth=2, maxiter=120, seed=5))
    result = solver.solve(problem)

    assert result.solver_name == "vqe"
    assert set(result.global_observables) >= {"Mz", "Mx", "ZZ_nn", "Mstag2", "Z_span", "energy"}
    assert "Mz_site" in result.site_observables
    assert "Mx_site" in result.site_observables
    assert len(result.site_observables["Mz_site"]) == 4
    assert len(result.site_observables["Mx_site"]) == 4
    assert result.statevector is not None
    assert result.metadata["method"] == "tfim_vqe"
    assert result.metadata["parameter_count"] == 6
    assert isinstance(result.metadata["energy_history"], list)


def test_tfim_vqe_energy_is_close_to_exact_reference() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    exact = ExactEDSolver().solve(problem)
    vqe = VQESolver(VQESettings(depth=3, maxiter=200, seed=7)).solve(problem)

    assert abs(vqe.energy - exact.energy) < 0.15
    assert abs(vqe.global_observables["ZZ_nn"] - exact.global_observables["ZZ_nn"]) < 0.2

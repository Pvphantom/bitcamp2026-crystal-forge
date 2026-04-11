from fastapi.testclient import TestClient

from app.domain.problem_spec import ProblemSpec
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from app.main import app


client = TestClient(app)


def test_tfim_exact_and_mean_field_solvers_return_expected_keys() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    exact = ExactEDSolver().solve(problem)
    approx = TFIMMeanFieldSolver().solve(problem)
    for result in (exact, approx):
        assert set(result.global_observables) >= {"Mz", "Mx", "ZZ_nn", "Mstag2", "Z_span", "energy"}
        assert "Mz_site" in result.site_observables
        assert len(result.site_observables["Mz_site"]) == 4


def test_generic_workflow_endpoint_supports_tfim_analysis() -> None:
    response = client.post(
        "/api/workflow/analyze",
        json={
            "model_family": "tfim",
            "Lx": 2,
            "Ly": 2,
            "parameters": {"J": 1.0, "h": 0.8, "g": 0.0},
            "qprobe_targets": ["Mz", "ZZ_nn", "Mstag2"],
            "qprobe_tolerance": 0.03,
            "qprobe_shots_per_group": 4000,
            "qprobe_readout_flip_prob": 0.02,
            "qprobe_seed": 7,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["model_family"] == "tfim"
    assert "vqe" in body["available_solvers"]
    assert body["selected_cheap_solver"] == "tfim_mean_field"
    assert body["selected_strong_solver"] == "vqe"
    assert body["workflow_decision"]["escalation_triggered"] is True
    assert body["workflow_decision"]["active_solver"] == "vqe"
    assert body["workflow_decision"]["measurement_mode"] == "quantum_follow_on"
    assert body["exact_solver"]["solver_name"] == "exact_ed"
    assert body["cheap_solver"]["solver_name"] == "tfim_mean_field"
    assert body["strong_solver"]["solver_name"] == "vqe"
    assert body["qprobe_exact"]["planning_state_solver"] == "vqe"
    assert body["qprobe_exact"]["oracle_reference_solver"] == "exact_ed"
    assert body["qprobe_adaptive"]["planning_state_solver"] == "vqe"
    assert body["qprobe_adaptive"]["oracle_reference_solver"] == "exact_ed"
    assert "trust" in body
    assert "qprobe_exact" in body
    assert "qprobe_adaptive" in body
    assert "Mz" in body["measurement_library"]["observables"]


def test_generic_workflow_hubbard_has_no_vqe_strong_solver() -> None:
    response = client.post(
        "/api/workflow/analyze",
        json={
            "model_family": "hubbard",
            "Lx": 2,
            "Ly": 2,
            "parameters": {"t": 1.0, "U": 4.0, "mu": 2.0},
            "qprobe_targets": ["D", "Ms2", "Cs_max"],
            "qprobe_tolerance": 0.03,
            "qprobe_shots_per_group": 2000,
            "qprobe_readout_flip_prob": 0.01,
            "qprobe_seed": 7,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["model_family"] == "hubbard"
    assert body["selected_cheap_solver"] == "mean_field"
    assert body["selected_strong_solver"] is None
    assert body["strong_solver"] is None
    assert body["workflow_decision"]["active_solver"] in {"mean_field", "exact_ed"}
    if body["workflow_decision"]["escalation_triggered"]:
        assert body["workflow_decision"]["measurement_mode"] == "oracle_fallback"
        assert body["workflow_decision"]["active_solver"] == "exact_ed"
        assert body["qprobe_exact"] is None
        assert body["qprobe_adaptive"] is None
    else:
        assert body["workflow_decision"]["measurement_mode"] == "not_needed"
        assert body["workflow_decision"]["active_solver"] == "mean_field"
        assert body["qprobe_exact"] is None
        assert body["qprobe_adaptive"] is None

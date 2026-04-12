from fastapi.testclient import TestClient

from app.api.routes import workflow_service
from app.domain.problem_spec import ProblemSpec
import app.services.workflow as workflow
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
    assert body["workflow_decision"]["active_solver"] in {"vqe", "exact_ed"}
    assert body["workflow_decision"]["measurement_mode"] in {"quantum_follow_on", "oracle_fallback"}
    assert body["workflow_decision"]["route_label"] in {"quantum_frontier", "scalable_classical", "uncertain"}
    assert body["exact_solver"]["solver_name"] == "exact_ed"
    assert body["cheap_solver"]["solver_name"] == "tfim_mean_field"
    assert body["strong_solver"]["solver_name"] == "vqe"
    assert body["qprobe_exact"]["planning_state_solver"] == "vqe"
    assert body["qprobe_exact"]["oracle_reference_solver"] == "exact_ed"
    assert body["qprobe_adaptive"]["planning_state_solver"] == "vqe"
    assert body["qprobe_adaptive"]["oracle_reference_solver"] == "exact_ed"
    assert "trust" in body
    assert "routing" in body
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
    assert body["workflow_decision"]["route_label"] in {"mean_field", "scalable_classical"}
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


def test_generic_workflow_uses_routing_overlay_when_available() -> None:
    original_hybrid_predict = workflow_service.hybrid_corrmap_inference.predict
    original_predict = workflow_service.routing_inference.predict
    original_runtime_intrinsic = workflow.analyze_runtime_intrinsic_corrmap
    workflow_service.hybrid_corrmap_inference.predict = lambda features: {
        "available": True,
        "model_path": "test-routing-model.pt",
        "label": "mean_field",
        "confidence": 0.9,
        "recommended_action": "use_mean_field",
        "candidate_scores": {"mean_field": 0.9, "scalable_classical": 0.1},
        "abstained": False,
        "abstain_reason": None,
        "intrinsic_label": "stable_classical",
        "intrinsic_score": 0.9,
    }
    workflow_service.routing_inference.predict = lambda features: None
    workflow.analyze_runtime_intrinsic_corrmap = lambda problem: type(
        "StubRuntimeIntrinsic",
        (),
        {
            "assessment": type("Assessment", (), {"label": "stable_classical", "score": 0.2, "reasons": []})(),
            "ansatz_disagreement": type(
                "Ansatz",
                (),
                {"max_abs_gap": 0.0, "energy_density_gap": 0.0},
            )(),
            "hysteresis": type(
                "Hysteresis",
                (),
                {"observable_gap_max": 0.0, "energy_density_gap": 0.0},
            )(),
        },
    )()
    try:
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
    finally:
        workflow_service.hybrid_corrmap_inference.predict = original_hybrid_predict
        workflow_service.routing_inference.predict = original_predict
        workflow.analyze_runtime_intrinsic_corrmap = original_runtime_intrinsic

    body = response.json()
    assert response.status_code == 200
    assert body["routing"]["route_label"] == "mean_field"
    assert body["routing"]["abstained"] is False
    assert body["routing"]["intrinsic_label"] == "stable_classical"
    assert body["workflow_decision"]["route_label"] == "mean_field"
    assert body["workflow_decision"]["active_solver"] == "tfim_mean_field"
    assert body["workflow_decision"]["measurement_mode"] == "not_needed"


def test_generic_workflow_hybrid_guard_blocks_mean_field_when_intrinsic_risk_is_high() -> None:
    original_hybrid_predict = workflow_service.hybrid_corrmap_inference.predict
    original_predict = workflow_service.routing_inference.predict
    original_runtime_intrinsic = workflow.analyze_runtime_intrinsic_corrmap
    workflow_service.hybrid_corrmap_inference.predict = lambda features: {
        "available": True,
        "model_path": "test-hybrid-model.pt",
        "label": "uncertain",
        "confidence": 0.92,
        "recommended_action": "abstain_or_collect_stronger_evidence",
        "candidate_scores": {"mean_field": 0.9, "scalable_classical": 0.1},
        "abstained": True,
        "abstain_reason": "intrinsic_risk_guard",
        "intrinsic_label": "frontier_or_uncertain",
        "intrinsic_score": 0.95,
    }
    workflow_service.routing_inference.predict = lambda features: None
    workflow.analyze_runtime_intrinsic_corrmap = lambda problem: type(
        "StubRuntimeIntrinsic",
        (),
        {
            "assessment": type(
                "Assessment",
                (),
                {"label": "frontier_or_uncertain", "score": 4.5, "reasons": ["ansatz_disagreement"]},
            )(),
            "ansatz_disagreement": type(
                "Ansatz",
                (),
                {"max_abs_gap": 0.2, "energy_density_gap": 0.1},
            )(),
            "hysteresis": type(
                "Hysteresis",
                (),
                {"observable_gap_max": 0.2, "energy_density_gap": 0.1},
            )(),
        },
    )()
    try:
        response = client.post(
            "/api/workflow/analyze",
            json={
                "model_family": "hubbard",
                "Lx": 2,
                "Ly": 2,
                "parameters": {"t": 1.0, "U": 4.0, "mu": 2.0},
            },
        )
    finally:
        workflow_service.hybrid_corrmap_inference.predict = original_hybrid_predict
        workflow_service.routing_inference.predict = original_predict
        workflow.analyze_runtime_intrinsic_corrmap = original_runtime_intrinsic

    body = response.json()
    assert response.status_code == 200
    assert body["routing"]["abstained"] is False
    assert body["routing"]["abstain_reason"] == "intrinsic_runtime_quantum_escalation"
    assert body["workflow_decision"]["route_label"] == "quantum_frontier"

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from app.domain.models import ObservablePauliTermRequest, ObservableTargetSpecRequest, QProbeRequest
from app.domain.problem_spec import ProblemSpec
from app.observables.registry import build_default_observable_registry
from app.observables.request_compiler import resolve_observable_requests
from app.optimization.measurement_plan import (
    search_adaptive_measurement_plan_for_problem,
    search_adaptive_measurement_plan_with_operator_map,
    search_minimal_measurement_plan_for_problem,
    search_minimal_measurement_plan_with_operator_map,
)
from app.physics.measurement_eval import NoiseModel
from app.services.game_state import HubbardGameStateService
from app.solvers.exact_ed import ExactEDSolver
from fastapi.testclient import TestClient
from app.main import app
import pytest


client = TestClient(app)


def _spec_from_operator(name: str, operator: SparsePauliOp) -> ObservableTargetSpecRequest:
    simplified = operator.simplify()
    return ObservableTargetSpecRequest(
        alias=name,
        pauli_terms=[
            ObservablePauliTermRequest(
                pauli=str(pauli),
                coeff_real=float(coeff.real),
                coeff_imag=float(coeff.imag),
            )
            for pauli, coeff in zip(simplified.paulis, simplified.coeffs, strict=True)
        ],
    )


def test_request_compiler_custom_operator_matches_named_registry_operator() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    named_targets, named_map = resolve_observable_requests(
        problem=problem,
        registry=registry,
        target_names=["D"],
        observable_specs=[],
    )
    custom_spec = _spec_from_operator("D_custom", named_map["D"])
    custom_targets, custom_map = resolve_observable_requests(
        problem=problem,
        registry=registry,
        observable_specs=[custom_spec],
    )
    assert named_targets == ("D",)
    assert custom_targets == ("D_custom",)
    assert custom_map["D_custom"].equiv(named_map["D"])


def test_exact_qprobe_plan_is_unchanged_for_equivalent_custom_operator() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    exact = ExactEDSolver(registry).solve(problem)
    operator = registry.operator("D", problem)
    custom_spec = _spec_from_operator("D_custom", operator)
    _, custom_map = resolve_observable_requests(problem=problem, registry=registry, observable_specs=[custom_spec])

    named = search_minimal_measurement_plan_for_problem(
        problem=problem,
        state=exact.statevector,
        target_observables=("D",),
        tolerance=0.03,
        shots_per_group=2000,
        noise_model=NoiseModel(),
        seed=7,
        registry=registry,
    )
    custom = search_minimal_measurement_plan_with_operator_map(
        state=exact.statevector,
        operator_map=custom_map,
        target_observables=("D_custom",),
        tolerance=0.03,
        shots_per_group=2000,
        noise_model=NoiseModel(),
        seed=7,
    )

    assert named.success == custom.success
    assert named.full_plan.bases == custom.full_plan.bases
    assert named.recommended_plan.bases == custom.recommended_plan.bases
    assert named.recommended_plan.cost == custom.recommended_plan.cost
    assert abs(named.exact["D"] - custom.exact["D_custom"]) < 1e-9


def test_adaptive_qprobe_plan_is_unchanged_for_equivalent_custom_operator() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    exact = ExactEDSolver(registry).solve(problem)
    operator = registry.operator("Ms2", problem)
    custom_spec = _spec_from_operator("Ms2_custom", operator)
    _, custom_map = resolve_observable_requests(problem=problem, registry=registry, observable_specs=[custom_spec])

    named = search_adaptive_measurement_plan_for_problem(
        problem=problem,
        state=exact.statevector,
        target_observables=("Ms2",),
        tolerance=0.03,
        shots_per_group=2000,
        noise_model=NoiseModel(),
        seed=11,
        registry=registry,
    )
    custom = search_adaptive_measurement_plan_with_operator_map(
        state=exact.statevector,
        operator_map=custom_map,
        target_observables=("Ms2_custom",),
        tolerance=0.03,
        shots_per_group=2000,
        noise_model=NoiseModel(),
        seed=11,
    )

    assert named.full_plan.bases == custom.full_plan.bases
    assert named.final_plan.bases == custom.final_plan.bases
    assert named.final_plan.cost == custom.final_plan.cost
    assert named.success == custom.success


def test_game_state_custom_operator_request_disables_ml_qprobe() -> None:
    service = HubbardGameStateService()
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=service.config.Lx, Ly=service.config.Ly, t=service.config.t, U=service.config.U, mu=service.config.mu)
    operator = registry.operator("D", problem)
    custom_spec = _spec_from_operator("D_custom", operator)
    response = service.recommend_qprobe_plan(
        QProbeRequest(
            targets=[],
            observable_specs=[custom_spec],
            tolerance=0.03,
            shots_per_group=2000,
            readout_flip_prob=0.0,
            seed=5,
        )
    )
    assert response.targets == ["D_custom"]
    assert response.ml_qprobe.available is False


def test_game_state_custom_operator_request_uses_general_ml_qprobe_when_available() -> None:
    service = HubbardGameStateService()
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=service.config.Lx, Ly=service.config.Ly, t=service.config.t, U=service.config.U, mu=service.config.mu)
    operator = registry.operator("D", problem)
    custom_spec = _spec_from_operator("D_custom", operator)

    class _StubGeneralEngine:
        model_path = "general-qprobe.pt"

        @staticmethod
        def predict(features):
            return {
                "available": True,
                "model_path": "general-qprobe.pt",
                "predicted_cost": 1,
                "predicted_success": True,
                "predicted_error": 0.01,
            }

    service.general_qprobe_inference = _StubGeneralEngine()
    response = service.recommend_qprobe_plan(
        QProbeRequest(
            targets=[],
            observable_specs=[custom_spec],
            tolerance=0.03,
            shots_per_group=2000,
            readout_flip_prob=0.0,
            seed=5,
        )
    )
    assert response.targets == ["D_custom"]
    assert response.ml_qprobe.available is True
    assert response.ml_qprobe.model_path == "general-qprobe.pt"


def test_workflow_accepts_custom_qprobe_operator_on_quantum_path() -> None:
    client = TestClient(app)
    tfim_problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    registry = build_default_observable_registry()
    custom_spec = _spec_from_operator("Mz_custom", registry.operator("Mz", tfim_problem))
    response = client.post(
        "/api/workflow/analyze",
        json={
            "model_family": "tfim",
            "Lx": 2,
            "Ly": 2,
            "parameters": {"J": 1.0, "h": 0.8, "g": 0.0},
            "qprobe_observable_specs": [custom_spec.model_dump()],
            "qprobe_tolerance": 0.03,
            "qprobe_shots_per_group": 4000,
            "qprobe_readout_flip_prob": 0.02,
            "qprobe_seed": 7,
        },
    )
    body = response.json()
    assert response.status_code == 200
    if body["qprobe_exact"] is not None:
        assert body["qprobe_exact"]["targets"] == ["Mz_custom"]
    if body["qprobe_adaptive"] is not None:
        assert body["qprobe_adaptive"]["targets"] == ["Mz_custom"]


def test_game_state_rejects_more_than_five_targets() -> None:
    service = HubbardGameStateService()
    with pytest.raises(Exception):
        service.recommend_qprobe_plan(
            QProbeRequest(
                targets=["D", "n", "Ms2", "K", "Cs_max", "D"],
                tolerance=0.03,
                shots_per_group=2000,
            )
        )


def test_api_rejects_too_many_qprobe_targets() -> None:
    response = client.post(
        "/api/qprobe/recommend-plan",
        json={
            "targets": ["D", "n", "Ms2", "K", "Cs_max", "D"],
            "tolerance": 0.03,
            "shots_per_group": 2000,
        },
    )
    assert response.status_code == 422


def test_api_rejects_custom_operator_with_large_support() -> None:
    response = client.post(
        "/api/qprobe/recommend-plan",
        json={
            "targets": [],
            "observable_specs": [
                {
                    "alias": "too_big",
                    "pauli_terms": [
                        {"pauli": "XXXXXXXX", "coeff_real": 1.0, "coeff_imag": 0.0},
                    ],
                }
            ],
            "tolerance": 0.03,
            "shots_per_group": 2000,
        },
    )
    assert response.status_code == 422
    assert "support size" in response.json()["detail"]


def test_api_rejects_custom_request_with_too_many_pauli_terms() -> None:
    specs = []
    for spec_idx in range(4):
        pauli_terms = []
        for term_idx in range(7):
            chars = ["I"] * 8
            a = (spec_idx + term_idx) % 8
            b = (spec_idx * 2 + term_idx + 1) % 8
            chars[a] = "Z" if term_idx % 2 == 0 else "X"
            chars[b] = "X" if term_idx % 3 == 0 else "Y"
            pauli_terms.append({"pauli": "".join(chars), "coeff_real": 0.1, "coeff_imag": 0.0})
        specs.append({"alias": f"too_many_terms_{spec_idx}", "pauli_terms": pauli_terms})
    response = client.post(
        "/api/qprobe/recommend-plan",
        json={
            "targets": [],
            "observable_specs": specs,
            "tolerance": 0.03,
            "shots_per_group": 2000,
        },
    )
    assert response.status_code == 422
    assert "Pauli terms total" in response.json()["detail"]

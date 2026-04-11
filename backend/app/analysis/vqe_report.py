from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import GenericProblemRequest
from app.services.workflow import WorkflowService


@dataclass(frozen=True)
class VQEReportScenario:
    name: str
    description: str
    payload: GenericProblemRequest


SCENARIOS = (
    VQEReportScenario(
        name="tfim_easy_safe",
        description="Cheap solver should remain reasonably trustworthy, so the workflow should avoid quantum escalation.",
        payload=GenericProblemRequest(
            model_family="tfim",
            Lx=2,
            Ly=2,
            parameters={"J": 0.1, "h": 0.5, "g": 1.0},
            qprobe_targets=["Mz", "Mx", "ZZ_nn"],
            qprobe_tolerance=0.03,
            qprobe_shots_per_group=4000,
            qprobe_readout_flip_prob=0.02,
            qprobe_seed=7,
        ),
    ),
    VQEReportScenario(
        name="tfim_quantum_escalation",
        description="Cheap solver becomes unreliable enough that the workflow escalates to VQE and turns on QProbe.",
        payload=GenericProblemRequest(
            model_family="tfim",
            Lx=2,
            Ly=2,
            parameters={"J": 1.0, "h": 0.8, "g": 0.0},
            qprobe_targets=["Mz", "ZZ_nn", "Mstag2"],
            qprobe_tolerance=0.03,
            qprobe_shots_per_group=4000,
            qprobe_readout_flip_prob=0.02,
            qprobe_seed=7,
        ),
    ),
)


def build_vqe_report(service: WorkflowService | None = None) -> dict[str, object]:
    workflow = service or WorkflowService()
    scenario_reports: dict[str, object] = {}
    escalated = 0
    qprobe_runs = 0
    qprobe_savings = []

    for scenario in SCENARIOS:
        result = workflow.analyze(scenario.payload)
        exact_energy = result.exact_solver.energy
        cheap_energy = result.cheap_solver.energy
        strong_energy = result.strong_solver.energy if result.strong_solver is not None else None
        cheap_gap = abs(cheap_energy - exact_energy)
        strong_gap = abs(strong_energy - exact_energy) if strong_energy is not None else None
        if result.workflow_decision.escalation_triggered:
            escalated += 1
        if result.qprobe_exact is not None:
            qprobe_runs += 1
            qprobe_savings.append(result.qprobe_exact.measurement_savings)

        scenario_reports[scenario.name] = {
            "description": scenario.description,
            "model_family": result.model_family,
            "parameters": result.parameters,
            "workflow_decision": result.workflow_decision.model_dump(),
            "trust": result.trust.model_dump(),
            "solver_summary": {
                "cheap_solver": result.cheap_solver.solver_name,
                "strong_solver": result.strong_solver.solver_name if result.strong_solver is not None else None,
                "exact_energy": exact_energy,
                "cheap_energy": cheap_energy,
                "strong_energy": strong_energy,
                "cheap_energy_gap": cheap_gap,
                "strong_energy_gap": strong_gap,
            },
            "qprobe": result.qprobe_exact.model_dump() if result.qprobe_exact is not None else None,
            "adaptive_qprobe": result.qprobe_adaptive.model_dump() if result.qprobe_adaptive is not None else None,
        }

    return {
        "summary": {
            "num_scenarios": len(SCENARIOS),
            "num_escalated": escalated,
            "num_qprobe_runs": qprobe_runs,
            "avg_qprobe_savings": (sum(qprobe_savings) / len(qprobe_savings)) if qprobe_savings else 0.0,
        },
        "scenarios": scenario_reports,
    }

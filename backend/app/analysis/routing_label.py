from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.schema import REFERENCE_QUALITY_LABELS, ROUTING_LABELS, SolverBenchmarkOutcome


MEAN_FIELD_SOLVERS = {"mean_field", "tfim_mean_field"}
SCALABLE_CLASSICAL_SOLVERS = set()
QUANTUM_SOLVERS = {"vqe"}
ROUTE_PRIORITY = {
    "mean_field": 0,
    "scalable_classical": 1,
    "quantum_frontier": 2,
    "uncertain": 3,
}
COST_CLASS_PRIORITY = {
    "cheap": 0,
    "moderate": 1,
    "expensive": 2,
    "frontier": 3,
}


@dataclass(frozen=True)
class RoutingLabelConfig:
    observable_tolerance: float = 0.08
    energy_tolerance: float = 0.25
    allow_weak_labels: bool = False


@dataclass(frozen=True)
class RoutingLabelDecision:
    route_label: str
    recommended_action: str
    chosen_solver: str | None
    eligible_solvers: list[str] = field(default_factory=list)
    rejected_solvers: dict[str, str] = field(default_factory=dict)
    label_source: str = ""
    reference_quality: str = "unknown"


def route_family_for_solver(solver_name: str) -> str:
    if solver_name in MEAN_FIELD_SOLVERS:
        return "mean_field"
    if solver_name in SCALABLE_CLASSICAL_SOLVERS:
        return "scalable_classical"
    if solver_name in QUANTUM_SOLVERS:
        return "quantum_frontier"
    if solver_name == "exact_ed":
        return "oracle_reference"
    return "scalable_classical"


def build_routing_label(
    solver_outcomes: dict[str, SolverBenchmarkOutcome],
    *,
    reference_solver: str,
    reference_quality: str,
    config: RoutingLabelConfig | None = None,
) -> RoutingLabelDecision:
    settings = config or RoutingLabelConfig()
    if reference_quality not in REFERENCE_QUALITY_LABELS:
        raise ValueError(f"Unsupported reference quality: {reference_quality}")
    if reference_solver not in solver_outcomes:
        raise ValueError("Reference solver must be included in routing outcomes")
    if reference_quality == "unknown":
        return _uncertain_decision(
            reason=f"reference={reference_solver};quality=unknown",
            reference_quality=reference_quality,
        )
    if reference_quality == "weak" and not settings.allow_weak_labels:
        return _uncertain_decision(
            reason=f"reference={reference_solver};quality=weak;policy=abstain",
            reference_quality=reference_quality,
        )
    if (
        reference_quality == "weak"
        and settings.allow_weak_labels
        and reference_solver in MEAN_FIELD_SOLVERS
    ):
        return RoutingLabelDecision(
            route_label="mean_field",
            recommended_action=_recommended_action("mean_field"),
            chosen_solver=reference_solver,
            eligible_solvers=[reference_solver],
            rejected_solvers={},
            label_source=f"reference={reference_solver};quality=weak;policy=allow_mean_field",
            reference_quality=reference_quality,
        )

    eligible: list[SolverBenchmarkOutcome] = []
    rejected: dict[str, str] = {}
    for name, outcome in solver_outcomes.items():
        if name == reference_solver:
            continue
        reason = _ineligible_reason(
            outcome,
            reference_solver=reference_solver,
            observable_tolerance=settings.observable_tolerance,
            energy_tolerance=settings.energy_tolerance,
        )
        if reason is None:
            eligible.append(outcome)
        else:
            rejected[name] = reason

    if not eligible:
        reference_family = route_family_for_solver(reference_solver)
        frontier_route = _fallback_frontier_route(solver_outcomes)
        if reference_family == "oracle_reference" and frontier_route is not None:
            return RoutingLabelDecision(
                route_label=frontier_route,
                recommended_action=_recommended_action(frontier_route),
                chosen_solver=None,
                eligible_solvers=[],
                rejected_solvers=rejected,
                label_source=(
                    f"reference={reference_solver};quality={reference_quality};"
                    f"observable_tol={settings.observable_tolerance};energy_tol={settings.energy_tolerance};"
                    f"fallback_route={frontier_route}"
                ),
                reference_quality=reference_quality,
            )
        return _uncertain_decision(
            reason=(
                f"reference={reference_solver};quality={reference_quality};"
                f"observable_tol={settings.observable_tolerance};energy_tol={settings.energy_tolerance};"
                "eligible=none"
            ),
            reference_quality=reference_quality,
            rejected_solvers=rejected,
        )

    chosen = min(eligible, key=_solver_rank)
    route_label = route_family_for_solver(chosen.solver_name)
    assert route_label in ROUTING_LABELS
    return RoutingLabelDecision(
        route_label=route_label,
        recommended_action=_recommended_action(route_label),
        chosen_solver=chosen.solver_name,
        eligible_solvers=sorted(outcome.solver_name for outcome in eligible),
        rejected_solvers=rejected,
        label_source=(
            f"reference={reference_solver};quality={reference_quality};"
            f"observable_tol={settings.observable_tolerance};energy_tol={settings.energy_tolerance};"
            f"chosen={chosen.solver_name}"
        ),
        reference_quality=reference_quality,
    )


def _ineligible_reason(
    outcome: SolverBenchmarkOutcome,
    *,
    reference_solver: str,
    observable_tolerance: float,
    energy_tolerance: float,
) -> str | None:
    if not outcome.succeeded:
        return "solve_failed"
    if outcome.max_abs_error is None:
        return "missing_observable_error"
    if outcome.max_abs_error > observable_tolerance:
        return f"observable_error>{observable_tolerance}"
    if outcome.energy_error is None:
        return "missing_energy_error"
    if outcome.energy_error > energy_tolerance:
        return f"energy_error>{energy_tolerance}"
    return None


def _solver_rank(outcome: SolverBenchmarkOutcome) -> tuple[int, int, float, str]:
    route_label = route_family_for_solver(outcome.solver_name)
    cost_class = outcome.cost_class or _default_cost_class(route_label)
    runtime = float("inf") if outcome.runtime_s is None else float(outcome.runtime_s)
    return (
        ROUTE_PRIORITY[route_label],
        COST_CLASS_PRIORITY.get(cost_class, COST_CLASS_PRIORITY["frontier"]),
        runtime,
        outcome.solver_name,
    )


def _default_cost_class(route_label: str) -> str:
    if route_label == "mean_field":
        return "cheap"
    if route_label == "scalable_classical":
        return "expensive"
    if route_label == "quantum_frontier":
        return "frontier"
    return "frontier"


def _recommended_action(route_label: str) -> str:
    if route_label == "mean_field":
        return "use_mean_field"
    if route_label == "scalable_classical":
        return "use_scalable_classical_solver"
    if route_label == "quantum_frontier":
        return "use_quantum_solver"
    return "abstain_or_collect_stronger_evidence"


def _fallback_frontier_route(solver_outcomes: dict[str, SolverBenchmarkOutcome]) -> str | None:
    if any(name in QUANTUM_SOLVERS and outcome.succeeded for name, outcome in solver_outcomes.items()):
        return "quantum_frontier"
    if any(name in MEAN_FIELD_SOLVERS for name in solver_outcomes):
        return "scalable_classical"
    return None


def _uncertain_decision(
    *,
    reason: str,
    reference_quality: str,
    rejected_solvers: dict[str, str] | None = None,
) -> RoutingLabelDecision:
    return RoutingLabelDecision(
        route_label="uncertain",
        recommended_action=_recommended_action("uncertain"),
        chosen_solver=None,
        eligible_solvers=[],
        rejected_solvers=rejected_solvers or {},
        label_source=reason,
        reference_quality=reference_quality,
    )

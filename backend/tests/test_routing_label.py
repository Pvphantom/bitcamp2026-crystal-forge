from app.analysis.routing_label import RoutingLabelConfig, build_routing_label
from app.ml.schema import SolverBenchmarkOutcome


def test_routing_label_prefers_mean_field_when_it_meets_tolerance() -> None:
    decision = build_routing_label(
        {
            "exact_ed": SolverBenchmarkOutcome(
                solver_name="exact_ed",
                family="oracle_reference",
                succeeded=True,
                runtime_s=2.0,
                max_abs_error=0.0,
                energy_error=0.0,
                cost_class="expensive",
            ),
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="mean_field",
                succeeded=True,
                runtime_s=0.01,
                max_abs_error=0.03,
                energy_error=0.1,
                cost_class="cheap",
            ),
            "vqe": SolverBenchmarkOutcome(
                solver_name="vqe",
                family="quantum_frontier",
                succeeded=True,
                runtime_s=20.0,
                max_abs_error=0.02,
                energy_error=0.05,
                cost_class="frontier",
            ),
        },
        reference_solver="exact_ed",
        reference_quality="strong",
    )
    assert decision.route_label == "mean_field"
    assert decision.chosen_solver == "mean_field"


def test_routing_label_escalates_to_quantum_frontier_when_mean_field_fails_and_quantum_is_available() -> None:
    decision = build_routing_label(
        {
            "exact_ed": SolverBenchmarkOutcome(
                solver_name="exact_ed",
                family="oracle_reference",
                succeeded=True,
                runtime_s=2.0,
                max_abs_error=0.0,
                energy_error=0.0,
            ),
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="mean_field",
                succeeded=True,
                runtime_s=0.01,
                max_abs_error=0.2,
                energy_error=0.6,
            ),
            "vqe": SolverBenchmarkOutcome(
                solver_name="vqe",
                family="quantum_frontier",
                succeeded=True,
                runtime_s=20.0,
                max_abs_error=0.03,
                energy_error=0.1,
            ),
        },
        reference_solver="exact_ed",
        reference_quality="strong",
        config=RoutingLabelConfig(observable_tolerance=0.08, energy_tolerance=0.25),
    )
    assert decision.route_label == "quantum_frontier"
    assert decision.chosen_solver == "vqe"
    assert decision.rejected_solvers["mean_field"].startswith("observable_error>")


def test_routing_label_uses_scalable_classical_fallback_when_mean_field_fails_and_no_quantum_is_available() -> None:
    decision = build_routing_label(
        {
            "exact_ed": SolverBenchmarkOutcome(
                solver_name="exact_ed",
                family="oracle_reference",
                succeeded=True,
                runtime_s=2.0,
                max_abs_error=0.0,
                energy_error=0.0,
            ),
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="mean_field",
                succeeded=True,
                runtime_s=0.01,
                max_abs_error=0.2,
                energy_error=0.6,
            ),
        },
        reference_solver="exact_ed",
        reference_quality="strong",
        config=RoutingLabelConfig(observable_tolerance=0.08, energy_tolerance=0.25),
    )
    assert decision.route_label == "scalable_classical"
    assert decision.chosen_solver is None


def test_routing_label_abstains_when_reference_quality_is_weak_and_policy_disallows_it() -> None:
    decision = build_routing_label(
        {
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="mean_field",
                succeeded=True,
                runtime_s=0.01,
                max_abs_error=0.0,
                energy_error=0.0,
            ),
        },
        reference_solver="mean_field",
        reference_quality="weak",
    )
    assert decision.route_label == "uncertain"
    assert decision.chosen_solver is None


def test_routing_label_can_allow_weak_labels_explicitly() -> None:
    decision = build_routing_label(
        {
            "mean_field": SolverBenchmarkOutcome(
                solver_name="mean_field",
                family="mean_field",
                succeeded=True,
                runtime_s=0.01,
                max_abs_error=0.0,
                energy_error=0.0,
            ),
        },
        reference_solver="mean_field",
        reference_quality="weak",
        config=RoutingLabelConfig(allow_weak_labels=True),
    )
    assert decision.route_label == "mean_field"
    assert decision.chosen_solver == "mean_field"

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.solver_compare import compare_solver_results
from app.domain.models import (
    AdaptiveMeasurementPlanResponse,
    AdaptiveMeasurementStepResponse,
    GenericAnalysisResponse,
    GenericProblemRequest,
    GenericSolverResultResponse,
    GenericTrustResponse,
    MeasurementGroupResponse,
    MeasurementLibraryResponse,
    MeasurementPlanResponse,
    MLQProbePredictionResponse,
    WorkflowDecisionResponse,
)
from app.domain.problem_spec import ProblemSpec
from app.observables.registry import build_default_observable_registry
from app.optimization.measurement_plan import (
    AdaptiveMeasurementStep,
    search_adaptive_measurement_plan_for_problem,
    search_minimal_measurement_plan_for_problem,
)
from app.physics.measurement_eval import NoiseModel
from app.physics.measurements import MeasurementGroup, build_measurement_library_for_problem, explain_stop_reason
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.registry import SolverRegistry
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from app.solvers.vqe import VQESolver


@dataclass
class WorkflowService:
    def __init__(self) -> None:
        self.observable_registry = build_default_observable_registry()
        self.solver_registry = SolverRegistry()
        self.solver_registry.register(ExactEDSolver(self.observable_registry))
        self.solver_registry.register(MeanFieldSolver())
        self.solver_registry.register(TFIMMeanFieldSolver())
        self.solver_registry.register(VQESolver(observable_registry=self.observable_registry))

    def analyze(self, payload: GenericProblemRequest) -> GenericAnalysisResponse:
        problem = self._build_problem(payload)
        available_solvers = self.solver_registry.available_for(problem)
        exact = self.solver_registry.get("exact_ed").solve(problem)
        cheap_solver_name = "mean_field" if problem.model_family == "hubbard" else "tfim_mean_field"
        cheap = self.solver_registry.get(cheap_solver_name).solve(problem)
        strong_solver_name = "vqe" if self.solver_registry.supports("vqe", problem) else None
        strong = self.solver_registry.get(strong_solver_name).solve(problem) if strong_solver_name else None
        comparison = compare_solver_results(problem, exact, cheap)
        needs_escalation = comparison.risk_label in {"warning", "unsafe"}
        escalation_triggered = needs_escalation
        fallback_exact_escalation = needs_escalation and strong is None
        if needs_escalation and strong is not None:
            active_solver = strong.solver_name
            measurement_mode = "quantum_follow_on"
            recommendation = (
                "Cheap solver looks unreliable here. Escalate to the quantum variational solver, then use QProbe to reduce readout cost."
            )
        elif fallback_exact_escalation:
            active_solver = exact.solver_name
            measurement_mode = "oracle_fallback"
            recommendation = (
                "Cheap solver looks unreliable here, but no strong quantum solver is registered for this model family yet. Fall back to the exact oracle on small systems."
            )
        else:
            active_solver = cheap.solver_name
            measurement_mode = "not_needed"
            recommendation = (
                "Cheap solver is probably sufficient here. No escalation is needed, so QProbe is only a benchmark tool for this case."
            )

        qprobe_exact_response = None
        qprobe_adaptive_response = None

        targets = tuple(
            payload.qprobe_targets or self.observable_registry.names_for_family(problem.model_family)[:3]
        )
        measurement_library = build_measurement_library_for_problem(problem, registry=self.observable_registry)
        if escalation_triggered and strong is not None:
            measurement_state_result = strong
            measurement_state_solver = measurement_state_result.solver_name
            qprobe_exact = search_minimal_measurement_plan_for_problem(
                problem=problem,
                state=measurement_state_result.statevector,
                target_observables=targets,
                tolerance=payload.qprobe_tolerance,
                shots_per_group=payload.qprobe_shots_per_group,
                noise_model=NoiseModel(readout_flip_prob=payload.qprobe_readout_flip_prob),
                seed=payload.qprobe_seed,
                registry=self.observable_registry,
            )
            qprobe_adaptive = search_adaptive_measurement_plan_for_problem(
                problem=problem,
                state=measurement_state_result.statevector,
                target_observables=targets,
                tolerance=payload.qprobe_tolerance,
                shots_per_group=payload.qprobe_shots_per_group,
                noise_model=NoiseModel(readout_flip_prob=payload.qprobe_readout_flip_prob),
                seed=payload.qprobe_seed,
                registry=self.observable_registry,
            )
            qprobe_exact_response = MeasurementPlanResponse(
                success=qprobe_exact.success,
                targets=list(qprobe_exact.target_observables),
                tolerance=qprobe_exact.tolerance,
                planning_state_solver=measurement_state_solver,
                oracle_reference_solver=exact.solver_name,
                full_cost=qprobe_exact.full_plan.cost,
                recommended_cost=qprobe_exact.recommended_plan.cost,
                measurement_savings=qprobe_exact.full_plan.cost - qprobe_exact.recommended_plan.cost,
                exact=qprobe_exact.exact,
                estimated=qprobe_exact.estimated,
                abs_error=qprobe_exact.abs_error,
                max_abs_error=qprobe_exact.max_abs_error,
                full_groups=[self._measurement_group_response(group) for group in qprobe_exact.full_plan.groups],
                recommended_groups=[self._measurement_group_response(group) for group in qprobe_exact.recommended_plan.groups],
                ml_qprobe=MLQProbePredictionResponse(available=False, model_path=""),
                message=qprobe_exact.message,
            )
            qprobe_adaptive_response = AdaptiveMeasurementPlanResponse(
                success=qprobe_adaptive.success,
                targets=list(qprobe_adaptive.target_observables),
                tolerance=qprobe_adaptive.tolerance,
                runtime_stop_rule=qprobe_adaptive.runtime_stop_rule,
                planning_state_solver=measurement_state_solver,
                oracle_reference_solver=exact.solver_name,
                full_cost=qprobe_adaptive.full_plan.cost,
                final_cost=qprobe_adaptive.final_plan.cost,
                measurement_savings=qprobe_adaptive.full_plan.cost - qprobe_adaptive.final_plan.cost,
                exact=qprobe_adaptive.exact,
                estimated=qprobe_adaptive.estimated,
                abs_error=qprobe_adaptive.abs_error,
                max_abs_error=qprobe_adaptive.max_abs_error,
                max_uncertainty=qprobe_adaptive.max_uncertainty,
                oracle_benchmark_within_tolerance=qprobe_adaptive.oracle_benchmark_within_tolerance,
                steps=[self._adaptive_step_response(step) for step in qprobe_adaptive.steps],
                message=f"{qprobe_adaptive.message} {explain_stop_reason(qprobe_adaptive.success, qprobe_adaptive.max_uncertainty, qprobe_adaptive.tolerance)}",
            )

        return GenericAnalysisResponse(
            model_family=problem.model_family,
            lattice={"Lx": problem.Lx, "Ly": problem.Ly},
            parameters=dict(problem.parameters.values),
            available_solvers=available_solvers,
            selected_cheap_solver=cheap_solver_name,
            selected_strong_solver=strong_solver_name,
            workflow_decision=WorkflowDecisionResponse(
                escalation_triggered=escalation_triggered,
                active_solver=active_solver,
                measurement_mode=measurement_mode,
                recommendation=recommendation,
            ),
            exact_solver=self._solver_result_response(exact),
            cheap_solver=self._solver_result_response(cheap),
            strong_solver=self._solver_result_response(strong) if strong is not None else None,
            trust=GenericTrustResponse(
                abs_error=comparison.abs_error,
                rel_error=comparison.rel_error,
                max_abs_error=comparison.max_abs_error,
                energy_error=comparison.energy_error,
                risk_label=comparison.risk_label,
                recommended_action=self._trust_action(comparison.risk_label),
            ),
            measurement_library=MeasurementLibraryResponse(
                observables={
                    name: [self._measurement_group_response(group, [name]) for group in groups]
                    for name, groups in measurement_library.items()
                }
            ),
            qprobe_exact=qprobe_exact_response,
            qprobe_adaptive=qprobe_adaptive_response,
        )

    @staticmethod
    def _build_problem(payload: GenericProblemRequest) -> ProblemSpec:
        params = payload.parameters
        if payload.model_family == "hubbard":
            return ProblemSpec.hubbard(
                Lx=payload.Lx,
                Ly=payload.Ly,
                t=float(params.get("t", 1.0)),
                U=float(params.get("U", 4.0)),
                mu=float(params.get("mu", 2.0)),
            )
        if payload.model_family == "tfim":
            return ProblemSpec.tfim(
                Lx=payload.Lx,
                Ly=payload.Ly,
                J=float(params.get("J", 1.0)),
                h=float(params.get("h", 1.0)),
                g=float(params.get("g", 0.0)),
            )
        raise ValueError(f"Unsupported model family: {payload.model_family}")

    @staticmethod
    def _trust_action(label: str) -> str:
        if label == "safe":
            return "cheap_solver_ok"
        if label == "warning":
            return "check_exact_or_stronger_solver"
        return "escalate_to_exact_or_advanced_method"

    @staticmethod
    def _bond_key(i: int, j: int) -> str:
        return f"{i}-{j}"

    def _solver_result_response(self, result) -> GenericSolverResultResponse:
        return GenericSolverResultResponse(
            solver_name=result.solver_name,
            energy=result.energy,
            observables={key: float(value) for key, value in result.global_observables.items()},
            site_observables={key: [float(v) for v in values] for key, values in result.site_observables.items()},
            bond_observables={self._bond_key(*bond): float(value) for bond, value in result.bond_observables.items()},
            metadata=result.metadata,
        )

    def _measurement_group_response(
        self,
        group: MeasurementGroup,
        supports_targets: list[str] | None = None,
    ) -> MeasurementGroupResponse:
        return MeasurementGroupResponse(
            name=group.name,
            basis=group.basis,
            basis_label=group.basis_label,
            explanation=group.plain_english,
            supports_targets=supports_targets or [],
            num_terms=group.num_terms,
            cost=group.cost,
        )

    def _adaptive_step_response(self, step: AdaptiveMeasurementStep) -> AdaptiveMeasurementStepResponse:
        return AdaptiveMeasurementStepResponse(
            step_index=step.step_index,
            chosen_group=self._measurement_group_response(step.chosen_group),
            current_cost=step.plan.cost,
            covered_targets=list(step.covered_targets),
            unresolved_targets=list(step.unresolved_targets),
            estimated=step.estimated,
            exact=step.exact,
            abs_error=step.abs_error,
            max_abs_error=step.max_abs_error,
            uncertainty=step.uncertainty,
            max_uncertainty=step.max_uncertainty,
        )

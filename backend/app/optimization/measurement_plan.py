"""Measurement-plan search for QProbe."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from app.physics.measurement_eval import NoiseModel, evaluate_measurement_groups
from app.physics.measurements import MeasurementGroup, build_measurement_library


@dataclass(frozen=True)
class MeasurementPlan:
    groups: tuple[MeasurementGroup, ...]

    @property
    def group_names(self) -> list[str]:
        return [group.name for group in self.groups]

    @property
    def bases(self) -> list[str]:
        return [group.basis for group in self.groups]

    @property
    def cost(self) -> int:
        return sum(group.cost for group in self.groups)


@dataclass(frozen=True)
class MeasurementPlanResult:
    success: bool
    target_observables: tuple[str, ...]
    tolerance: float
    full_plan: MeasurementPlan
    recommended_plan: MeasurementPlan
    exact: dict[str, float]
    estimated: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    message: str


@dataclass(frozen=True)
class AdaptiveMeasurementStep:
    step_index: int
    chosen_group: MeasurementGroup
    plan: MeasurementPlan
    covered_targets: tuple[str, ...]
    unresolved_targets: tuple[str, ...]
    estimated: dict[str, float]
    exact: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    uncertainty: dict[str, float]
    max_uncertainty: float


@dataclass(frozen=True)
class AdaptiveMeasurementPlanResult:
    success: bool
    target_observables: tuple[str, ...]
    tolerance: float
    full_plan: MeasurementPlan
    final_plan: MeasurementPlan
    steps: tuple[AdaptiveMeasurementStep, ...]
    runtime_stop_rule: str
    exact: dict[str, float]
    estimated: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    max_uncertainty: float
    oracle_benchmark_within_tolerance: bool
    message: str


def _merged_groups_for_targets(
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
) -> list[MeasurementGroup]:
    by_basis: dict[str, list[MeasurementGroup]] = {}
    for name in target_observables:
        for group in measurement_library[name]:
            by_basis.setdefault(group.basis, []).append(group)

    merged = []
    for basis, basis_groups in sorted(by_basis.items()):
        term_map: dict[str, complex] = {}
        for group in basis_groups:
            for term in group.terms:
                term_map[term.pauli] = term_map.get(term.pauli, 0.0j) + term.coeff
        merged.append(
            MeasurementGroup(
                name=f"merged:{basis}",
                basis=basis,
                terms=tuple(
                    type(basis_groups[0].terms[0])(pauli=pauli, coeff=coeff)
                    for pauli, coeff in sorted(term_map.items())
                    if abs(coeff) > 1e-12
                ),
            )
        )
    return merged


def _group_support_map(
    merged_groups: list[MeasurementGroup],
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
) -> dict[str, set[str]]:
    target_terms = {
        observable: {term.pauli for group in measurement_library[observable] for term in group.terms}
        for observable in target_observables
    }
    support = {}
    for group in merged_groups:
        group_terms = {term.pauli for term in group.terms}
        support[group.name] = {
            observable for observable, paulis in target_terms.items() if group_terms & paulis
        }
    return support


def _evaluate_plan(
    state: np.ndarray,
    plan: MeasurementPlan,
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
    *,
    shots_per_group: int,
    noise_model: NoiseModel,
    seed: int | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    _, term_expectations = evaluate_measurement_groups(
        state,
        list(plan.groups),
        shots_per_group=shots_per_group,
        noise_model=noise_model,
        seed=seed,
    )
    estimated: dict[str, float] = {}
    exact: dict[str, float] = {}
    abs_error: dict[str, float] = {}
    for observable in target_observables:
        exact_value = 0.0
        estimated_value = 0.0
        for group in measurement_library[observable]:
            for term in group.terms:
                exact_value += float(term.coeff.real) * _exact_pauli_expectation(state, term.pauli)
                estimated_value += float(term.coeff.real) * term_expectations.get(term.pauli, 0.0)
        estimated[observable] = estimated_value
        exact[observable] = exact_value
        abs_error[observable] = abs(estimated_value - exact_value)
    return estimated, exact, abs_error


def _estimate_uncertainty(
    state: np.ndarray,
    plan: MeasurementPlan,
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
    *,
    shots_per_group: int,
    noise_model: NoiseModel,
    seed: int | None,
    bootstrap_reps: int = 5,
) -> tuple[dict[str, float], float]:
    estimates_per_observable = {name: [] for name in target_observables}
    base_seed = 0 if seed is None else seed
    for rep in range(bootstrap_reps):
        estimated, _, _ = _evaluate_plan(
            state,
            plan,
            measurement_library,
            target_observables,
            shots_per_group=shots_per_group,
            noise_model=noise_model,
            seed=base_seed + rep + 1,
        )
        for name in target_observables:
            estimates_per_observable[name].append(estimated[name])
    uncertainty = {
        name: float(np.std(values))
        for name, values in estimates_per_observable.items()
    }
    return uncertainty, max(uncertainty.values()) if uncertainty else 0.0


def _covered_and_unresolved_targets(
    selected: list[MeasurementGroup],
    support_map: dict[str, set[str]],
    target_observables: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    covered = set().union(*(support_map[group.name] for group in selected)) if selected else set()
    unresolved = tuple(name for name in target_observables if name not in covered)
    return tuple(name for name in target_observables if name in covered), unresolved


def evaluate_adaptive_measurement_plan_oracle(
    *,
    state: np.ndarray,
    plan: MeasurementPlan,
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
    shots_per_group: int,
    noise_model: NoiseModel,
    seed: int | None,
    bootstrap_reps: int = 5,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, dict[str, float], float]:
    estimated, exact, abs_error = _evaluate_plan(
        state,
        plan,
        measurement_library,
        target_observables,
        shots_per_group=shots_per_group,
        noise_model=noise_model,
        seed=seed,
    )
    uncertainty, max_uncertainty = _estimate_uncertainty(
        state,
        plan,
        measurement_library,
        target_observables,
        shots_per_group=shots_per_group,
        noise_model=noise_model,
        seed=seed,
        bootstrap_reps=bootstrap_reps,
    )
    max_abs_error = max(abs_error.values()) if abs_error else 0.0
    return estimated, exact, abs_error, max_abs_error, uncertainty, max_uncertainty


def _exact_pauli_expectation(state: np.ndarray, pauli: str) -> float:
    from qiskit.quantum_info import SparsePauliOp

    return float(np.vdot(state, SparsePauliOp.from_list([(pauli, 1.0)]).to_matrix() @ state).real)


def search_minimal_measurement_plan(
    *,
    Lx: int,
    Ly: int,
    t: float,
    state: np.ndarray,
    target_observables: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> MeasurementPlanResult:
    measurement_library = build_measurement_library(Lx, Ly, t)
    merged_groups = _merged_groups_for_targets(measurement_library, target_observables)
    full_plan = MeasurementPlan(groups=tuple(merged_groups))
    noise = noise_model or NoiseModel()

    best_plan = full_plan
    best_estimated, best_exact, best_errors = _evaluate_plan(
        state,
        full_plan,
        measurement_library,
        target_observables,
        shots_per_group=shots_per_group,
        noise_model=noise,
        seed=seed,
    )
    best_max_error = max(best_errors.values()) if best_errors else 0.0

    for subset_size in range(1, len(merged_groups) + 1):
        passing_candidates: list[tuple[MeasurementPlan, dict[str, float], dict[str, float], dict[str, float]]] = []
        for subset in combinations(merged_groups, subset_size):
            plan = MeasurementPlan(groups=tuple(subset))
            estimated, exact, abs_error = _evaluate_plan(
                state,
                plan,
                measurement_library,
                target_observables,
                shots_per_group=shots_per_group,
                noise_model=noise,
                seed=seed,
            )
            if max(abs_error.values()) <= tolerance:
                passing_candidates.append((plan, estimated, exact, abs_error))
        if passing_candidates:
            passing_candidates.sort(key=lambda item: (item[0].cost, max(item[3].values()), item[0].group_names))
            best_plan, best_estimated, best_exact, best_errors = passing_candidates[0]
            best_max_error = max(best_errors.values()) if best_errors else 0.0
            return MeasurementPlanResult(
                success=True,
                target_observables=target_observables,
                tolerance=tolerance,
                full_plan=full_plan,
                recommended_plan=best_plan,
                exact=best_exact,
                estimated=best_estimated,
                abs_error=best_errors,
                max_abs_error=best_max_error,
                message="Found a compressed measurement plan within tolerance.",
            )

    return MeasurementPlanResult(
        success=False,
        target_observables=target_observables,
        tolerance=tolerance,
        full_plan=full_plan,
        recommended_plan=full_plan,
        exact=best_exact,
        estimated=best_estimated,
        abs_error=best_errors,
        max_abs_error=best_max_error,
        message="No compressed plan met the requested tolerance; returning the full plan.",
    )


def search_adaptive_measurement_plan(
    *,
    Lx: int,
    Ly: int,
    t: float,
    state: np.ndarray,
    target_observables: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    bootstrap_reps: int = 5,
) -> AdaptiveMeasurementPlanResult:
    measurement_library = build_measurement_library(Lx, Ly, t)
    merged_groups = _merged_groups_for_targets(measurement_library, target_observables)
    support_map = _group_support_map(merged_groups, measurement_library, target_observables)
    full_plan = MeasurementPlan(groups=tuple(merged_groups))
    noise = noise_model or NoiseModel()

    selected: list[MeasurementGroup] = []
    remaining = list(merged_groups)
    steps: list[AdaptiveMeasurementStep] = []
    final_estimated: dict[str, float] = {}
    final_exact: dict[str, float] = {}
    final_errors: dict[str, float] = {}
    final_max_error = float("inf")
    final_max_uncertainty = float("inf")

    while remaining:
        _, unresolved_targets = _covered_and_unresolved_targets(selected, support_map, target_observables)
        unresolved = set(unresolved_targets)
        remaining.sort(
            key=lambda group: (
                -len(support_map[group.name] & unresolved),
                -group.num_terms,
                group.basis,
            )
        )
        chosen_group = remaining[0]
        remaining.remove(chosen_group)
        selected.append(chosen_group)
        plan = MeasurementPlan(groups=tuple(selected))
        estimated, exact, abs_error, max_abs_error, uncertainty, max_uncertainty = (
            evaluate_adaptive_measurement_plan_oracle(
                state=state,
                plan=plan,
                measurement_library=measurement_library,
                target_observables=target_observables,
                shots_per_group=shots_per_group,
                noise_model=noise,
                seed=seed,
                bootstrap_reps=bootstrap_reps,
            )
        )
        covered_targets, unresolved_targets = _covered_and_unresolved_targets(
            selected,
            support_map,
            target_observables,
        )
        step = AdaptiveMeasurementStep(
            step_index=len(selected),
            chosen_group=chosen_group,
            plan=plan,
            covered_targets=covered_targets,
            unresolved_targets=unresolved_targets,
            estimated=estimated,
            exact=exact,
            abs_error=abs_error,
            max_abs_error=max_abs_error,
            uncertainty=uncertainty,
            max_uncertainty=max_uncertainty,
        )
        steps.append(step)
        final_estimated, final_exact, final_errors = estimated, exact, abs_error
        final_max_error = max_abs_error
        final_max_uncertainty = max_uncertainty
        runtime_confidence_bound = max_uncertainty + noise.readout_flip_prob
        if (
            not unresolved_targets
            and runtime_confidence_bound <= tolerance
            and len(selected) < len(merged_groups)
        ):
            return AdaptiveMeasurementPlanResult(
                success=True,
                target_observables=target_observables,
                tolerance=tolerance,
                full_plan=full_plan,
                final_plan=plan,
                steps=tuple(steps),
                runtime_stop_rule="coverage+uncertainty+noise-floor",
                exact=final_exact,
                estimated=final_estimated,
                abs_error=final_errors,
                max_abs_error=final_max_error,
                max_uncertainty=final_max_uncertainty,
                oracle_benchmark_within_tolerance=final_max_error <= tolerance,
                message="Adaptive QProbe stopped after it had covered all requested targets and the runtime uncertainty plus known readout-noise floor was below tolerance.",
            )

    return AdaptiveMeasurementPlanResult(
        success=False,
        target_observables=target_observables,
        tolerance=tolerance,
        full_plan=full_plan,
        final_plan=MeasurementPlan(groups=tuple(selected)),
        steps=tuple(steps),
        runtime_stop_rule="coverage+uncertainty+noise-floor",
        exact=final_exact,
        estimated=final_estimated,
        abs_error=final_errors,
        max_abs_error=final_max_error,
        max_uncertainty=final_max_uncertainty,
        oracle_benchmark_within_tolerance=final_max_error <= tolerance,
        message="Adaptive QProbe used every available measurement group before it could make a cost-saving runtime stop.",
    )

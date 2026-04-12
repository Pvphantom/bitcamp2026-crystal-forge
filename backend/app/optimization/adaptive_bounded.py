from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from app.optimization.measurement_plan import (
    AdaptiveMeasurementPlanResult,
    AdaptiveMeasurementStep,
    MeasurementPlan,
    evaluate_adaptive_measurement_plan_oracle,
    _covered_and_unresolved_targets,
    _group_support_map,
    _merged_groups_for_targets,
)
from app.physics.measurement_eval import NoiseModel
from app.physics.measurements import MeasurementGroup, build_measurement_library_from_operator_map


@dataclass(frozen=True)
class BoundedAdaptiveMetadata:
    nodes_visited: int
    node_budget: int
    certified_lower_bound: int
    best_cost_found: int
    beam_width: int


def search_bounded_adaptive_plan_with_operator_map(
    *,
    state: np.ndarray,
    operator_map: dict[str, object],
    target_observables: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    bootstrap_reps: int = 5,
    node_budget: int = 128,
) -> tuple[AdaptiveMeasurementPlanResult, BoundedAdaptiveMetadata]:
    measurement_library = build_measurement_library_from_operator_map(operator_map)
    merged_groups = _merged_groups_for_targets(measurement_library, target_observables)
    support_map = _group_support_map(merged_groups, measurement_library, target_observables)
    full_plan = MeasurementPlan(groups=tuple(merged_groups))
    noise = noise_model or NoiseModel()

    target_index = {name: idx for idx, name in enumerate(target_observables)}
    group_masks = {
        group.name: sum(1 << target_index[name] for name in support_map[group.name])
        for group in merged_groups
    }
    all_mask = (1 << len(target_observables)) - 1
    min_cover_dp = _min_cover_dp(group_masks.values(), all_mask)
    target_weight_map, group_target_weight = _weight_maps(
        measurement_library=measurement_library,
        merged_groups=merged_groups,
        target_observables=target_observables,
    )
    beam_width = max(8, min(len(merged_groups), max(8, node_budget // 8, 32 if len(merged_groups) <= 16 else 16)))

    greedy_result = _greedy_baseline(
        state=state,
        merged_groups=merged_groups,
        support_map=support_map,
        measurement_library=measurement_library,
        target_observables=target_observables,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        noise=noise,
        seed=seed,
        bootstrap_reps=bootstrap_reps,
    )
    best_plan = greedy_result.final_plan
    best_steps = list(greedy_result.steps)
    best_payload = {
        "estimated": greedy_result.estimated,
        "exact": greedy_result.exact,
        "abs_error": greedy_result.abs_error,
        "max_abs_error": greedy_result.max_abs_error,
        "max_uncertainty": greedy_result.max_uncertainty,
        "success": greedy_result.oracle_benchmark_within_tolerance,
    }
    best_cost = best_plan.cost

    nodes_visited = 0
    group_index = {group.name: idx for idx, group in enumerate(merged_groups)}
    progress_seen: dict[tuple[int, int], float] = {}
    frontier: list[tuple[float, float, int, int, tuple[int, ...]]] = []
    heapq.heappush(frontier, (float(min_cover_dp[all_mask]), -0.0, 0, 0, tuple()))

    while frontier and nodes_visited < node_budget:
        _, neg_progress, covered_mask, used_mask, selected_indices = heapq.heappop(frontier)
        selected = [merged_groups[idx] for idx in selected_indices]
        nodes_visited += 1
        lb_additional = min_cover_dp[all_mask ^ covered_mask]
        if len(selected) + lb_additional >= best_cost:
            continue
        if covered_mask == all_mask:
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
            if max_abs_error <= tolerance and plan.cost < best_cost:
                best_cost = plan.cost
                best_plan = plan
                best_payload = {
                    "estimated": estimated,
                    "exact": exact,
                    "abs_error": abs_error,
                    "max_abs_error": max_abs_error,
                    "max_uncertainty": max_uncertainty,
                    "success": True,
                }
                best_steps = _rebuild_steps(
                    selected_groups=list(selected),
                    support_map=support_map,
                    measurement_library=measurement_library,
                    target_observables=target_observables,
                    state=state,
                    shots_per_group=shots_per_group,
                    noise=noise,
                    seed=seed,
                    bootstrap_reps=bootstrap_reps,
                )
            continue

        remaining = [group for idx, group in enumerate(merged_groups) if not (used_mask & (1 << idx))]
        unresolved_mask = all_mask ^ covered_mask
        ordered = sorted(
            remaining,
            key=lambda group: (
                -_weighted_gain(
                    group_name=group.name,
                    uncovered_mask=unresolved_mask,
                    group_masks=group_masks,
                    target_observables=target_observables,
                    group_target_weight=group_target_weight,
                    target_weight_map=target_weight_map,
                ),
                -_bitcount(group_masks[group.name] & unresolved_mask),
                -group.num_terms,
                group.basis,
            ),
        )[:beam_width]

        for group in ordered:
            idx = group_index[group.name]
            new_used_mask = used_mask | (1 << idx)
            new_covered_mask = covered_mask | group_masks[group.name]
            progress = _covered_weight_progress(
                used_mask=new_used_mask,
                merged_groups=merged_groups,
                group_target_weight=group_target_weight,
                target_weight_map=target_weight_map,
                target_observables=target_observables,
            )
            signature = (_bitcount(new_used_mask), new_covered_mask)
            if progress_seen.get(signature, -1.0) >= progress:
                continue
            progress_seen[signature] = progress
            optimistic = _bitcount(new_used_mask) + min_cover_dp[all_mask ^ new_covered_mask]
            if optimistic >= best_cost:
                continue
            heapq.heappush(
                frontier,
                (
                    float(optimistic),
                    -progress,
                    new_covered_mask,
                    new_used_mask,
                    selected_indices + (idx,),
                ),
            )

    result = AdaptiveMeasurementPlanResult(
        success=best_payload["success"],
        target_observables=target_observables,
        tolerance=tolerance,
        full_plan=full_plan,
        final_plan=best_plan,
        steps=tuple(best_steps),
        runtime_stop_rule=f"bounded-search(node_budget={node_budget})",
        exact=best_payload["exact"],
        estimated=best_payload["estimated"],
        abs_error=best_payload["abs_error"],
        max_abs_error=best_payload["max_abs_error"],
        max_uncertainty=best_payload["max_uncertainty"],
        oracle_benchmark_within_tolerance=best_payload["max_abs_error"] <= tolerance,
        message="Bounded adaptive search returned the best tolerance-satisfying plan found within the search budget.",
    )
    meta = BoundedAdaptiveMetadata(
        nodes_visited=nodes_visited,
        node_budget=node_budget,
        certified_lower_bound=min_cover_dp[all_mask],
        best_cost_found=best_cost,
        beam_width=beam_width,
    )
    return result, meta


def _min_cover_dp(group_masks, all_mask: int) -> list[int]:
    inf = 10**9
    dp = [inf] * (all_mask + 1)
    dp[0] = 0
    masks = list(group_masks)
    for mask in range(all_mask + 1):
        if dp[mask] >= inf:
            continue
        for group_mask in masks:
            new_mask = mask | group_mask
            if dp[new_mask] > dp[mask] + 1:
                dp[new_mask] = dp[mask] + 1
    rem = [inf] * (all_mask + 1)
    for covered in range(all_mask + 1):
        rem[all_mask ^ covered] = dp[all_mask] if covered == 0 else dp[all_mask]  # placeholder for shape
    # return direct mapping from uncovered mask -> min groups to cover it
    uncovered_dp = [inf] * (all_mask + 1)
    uncovered_dp[0] = 0
    for mask in range(all_mask + 1):
        if uncovered_dp[mask] >= inf:
            continue
        for group_mask in masks:
            new_mask = mask | group_mask
            if uncovered_dp[new_mask] > uncovered_dp[mask] + 1:
                uncovered_dp[new_mask] = uncovered_dp[mask] + 1
    out = [0] * (all_mask + 1)
    for uncovered in range(all_mask + 1):
        out[uncovered] = uncovered_dp[uncovered]
    return out


def _bitcount(value: int) -> int:
    return int(value.bit_count())


def _weight_maps(
    *,
    measurement_library: dict[str, list[MeasurementGroup]],
    merged_groups: list[MeasurementGroup],
    target_observables: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    target_weight_map: dict[str, float] = {}
    group_target_weight: dict[str, dict[str, float]] = {group.name: {} for group in merged_groups}
    for target in target_observables:
        total = 0.0
        basis_weight: dict[str, float] = {}
        for group in measurement_library[target]:
            basis_weight[group.basis] = basis_weight.get(group.basis, 0.0) + sum(abs(term.coeff) for term in group.terms)
        total = sum(basis_weight.values())
        target_weight_map[target] = max(total, 1e-12)
        for merged_group in merged_groups:
            group_target_weight[merged_group.name][target] = basis_weight.get(merged_group.basis, 0.0)
    return target_weight_map, group_target_weight


def _weighted_gain(
    *,
    group_name: str,
    uncovered_mask: int,
    group_masks: dict[str, int],
    target_observables: tuple[str, ...],
    group_target_weight: dict[str, dict[str, float]],
    target_weight_map: dict[str, float],
) -> float:
    mask = group_masks[group_name] & uncovered_mask
    gain = 0.0
    for idx, target in enumerate(target_observables):
        if mask & (1 << idx):
            gain += group_target_weight[group_name][target] / target_weight_map[target]
    return gain


def _covered_weight_progress(
    *,
    used_mask: int,
    merged_groups: list[MeasurementGroup],
    group_target_weight: dict[str, dict[str, float]],
    target_weight_map: dict[str, float],
    target_observables: tuple[str, ...],
) -> float:
    covered = {target: 0.0 for target in target_observables}
    for idx, group in enumerate(merged_groups):
        if not (used_mask & (1 << idx)):
            continue
        weights = group_target_weight[group.name]
        for target in target_observables:
            covered[target] += weights[target]
    return sum(min(1.0, covered[target] / target_weight_map[target]) for target in target_observables)


def _greedy_baseline(
    *,
    state: np.ndarray,
    merged_groups: list[MeasurementGroup],
    support_map: dict[str, set[str]],
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise: NoiseModel,
    seed: int | None,
    bootstrap_reps: int,
) -> AdaptiveMeasurementPlanResult:
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
        chosen_group = remaining.pop(0)
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
        covered_targets, unresolved_targets = _covered_and_unresolved_targets(selected, support_map, target_observables)
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
        if not unresolved_targets and runtime_confidence_bound <= tolerance and len(selected) < len(merged_groups):
            return AdaptiveMeasurementPlanResult(
                success=True,
                target_observables=target_observables,
                tolerance=tolerance,
                full_plan=MeasurementPlan(groups=tuple(merged_groups)),
                final_plan=plan,
                steps=tuple(steps),
                runtime_stop_rule="greedy-coverage+uncertainty",
                exact=final_exact,
                estimated=final_estimated,
                abs_error=final_errors,
                max_abs_error=final_max_error,
                max_uncertainty=final_max_uncertainty,
                oracle_benchmark_within_tolerance=final_max_error <= tolerance,
                message="Greedy adaptive baseline stopped after coverage and uncertainty conditions were met.",
            )
    return AdaptiveMeasurementPlanResult(
        success=False,
        target_observables=target_observables,
        tolerance=tolerance,
        full_plan=MeasurementPlan(groups=tuple(merged_groups)),
        final_plan=MeasurementPlan(groups=tuple(selected)),
        steps=tuple(steps),
        runtime_stop_rule="greedy-coverage+uncertainty",
        exact=final_exact,
        estimated=final_estimated,
        abs_error=final_errors,
        max_abs_error=final_max_error,
        max_uncertainty=final_max_uncertainty,
        oracle_benchmark_within_tolerance=final_max_error <= tolerance,
        message="Greedy adaptive baseline used every available group.",
    )


def _rebuild_steps(
    *,
    selected_groups: list[MeasurementGroup],
    support_map: dict[str, set[str]],
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
    state: np.ndarray,
    shots_per_group: int,
    noise: NoiseModel,
    seed: int | None,
    bootstrap_reps: int,
) -> list[AdaptiveMeasurementStep]:
    steps: list[AdaptiveMeasurementStep] = []
    chosen: list[MeasurementGroup] = []
    for group in selected_groups:
        chosen.append(group)
        plan = MeasurementPlan(groups=tuple(chosen))
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
            chosen,
            support_map,
            target_observables,
        )
        steps.append(
            AdaptiveMeasurementStep(
                step_index=len(chosen),
                chosen_group=group,
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
        )
    return steps

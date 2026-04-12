from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.analysis.qprobe_request_budget import validate_qprobe_request_budget
from app.analysis.synthetic_operator_families_generalized import (
    GENERALIZED_OPERATOR_FAMILIES,
    HARD_OPERATOR_FAMILIES,
    build_generalized_synthetic_operator_bundle,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR
from app.optimization.adaptive_bounded import search_bounded_adaptive_plan_with_operator_map
from app.optimization.measurement_plan import (
    _merged_groups_for_targets,
    search_adaptive_measurement_plan_with_operator_map,
    search_minimal_measurement_plan_with_operator_map,
)
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.measurements import build_measurement_library_from_operator_map
from app.physics.tfim import build_tfim_hamiltonian


HUBBARD_POINTS = [
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 0.5, "mu": 0.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0},
]

TFIM_POINTS = [
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 0.8, "g": 0.0},
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 1.8, "g": 0.5},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "qprobe_adaptive_stack_report.json")
    parser.add_argument("--node-budget", type=int, default=64)
    parser.add_argument("--family-scope", choices=("all", "hard"), default="hard")
    parser.add_argument("--min-num-targets", type=int, default=2)
    parser.add_argument("--max-num-targets", type=int, default=4)
    parser.add_argument("--exact-max-full-cost", type=int, default=6)
    parser.add_argument("--base-seeds", type=int, nargs="*", default=[5, 11, 19])
    parser.add_argument("--hard-extra-seeds", type=int, nargs="*", default=[23, 29, 31, 37])
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--shots-per-group", type=int, default=2000)
    parser.add_argument("--readout-flip-prob", type=float, default=0.02)
    args = parser.parse_args()

    report = benchmark_stack(
        node_budget=args.node_budget,
        family_scope=args.family_scope,
        min_num_targets=args.min_num_targets,
        max_num_targets=args.max_num_targets,
        exact_max_full_cost=args.exact_max_full_cost,
        base_seeds=list(args.base_seeds),
        hard_extra_seeds=list(args.hard_extra_seeds),
        tolerance=args.tolerance,
        shots_per_group=args.shots_per_group,
        readout_flip_prob=args.readout_flip_prob,
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def benchmark_stack(
    *,
    node_budget: int,
    family_scope: str,
    min_num_targets: int,
    max_num_targets: int,
    exact_max_full_cost: int,
    base_seeds: list[int],
    hard_extra_seeds: list[int],
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    families = GENERALIZED_OPERATOR_FAMILIES if family_scope == "all" else tuple(sorted(HARD_OPERATOR_FAMILIES))
    noise_model = NoiseModel(readout_flip_prob=readout_flip_prob)

    for point in HUBBARD_POINTS:
        rows.extend(
            _benchmark_problem(
                problem=ProblemSpec.hubbard(**point),
                families=families,
                min_num_targets=min_num_targets,
                max_num_targets=max_num_targets,
                base_seeds=base_seeds,
                hard_extra_seeds=hard_extra_seeds,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                exact_max_full_cost=exact_max_full_cost,
                node_budget=node_budget,
            )
        )
    for point in TFIM_POINTS:
        rows.extend(
            _benchmark_problem(
                problem=ProblemSpec.tfim(**point),
                families=families,
                min_num_targets=min_num_targets,
                max_num_targets=max_num_targets,
                base_seeds=base_seeds,
                hard_extra_seeds=hard_extra_seeds,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                exact_max_full_cost=exact_max_full_cost,
                node_budget=node_budget,
            )
        )

    report = _summarize(rows)
    report.update(
        {
            "node_budget": node_budget,
            "family_scope": family_scope,
            "min_num_targets": min_num_targets,
            "max_num_targets": max_num_targets,
            "exact_max_full_cost": exact_max_full_cost,
            "tolerance": tolerance,
            "shots_per_group": shots_per_group,
            "readout_flip_prob": readout_flip_prob,
        }
    )
    return report


def _benchmark_problem(
    *,
    problem: ProblemSpec,
    families: tuple[str, ...],
    min_num_targets: int,
    max_num_targets: int,
    base_seeds: list[int],
    hard_extra_seeds: list[int],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel,
    exact_max_full_cost: int,
    node_budget: int,
) -> list[dict[str, object]]:
    state = _ground_state(problem)
    rows: list[dict[str, object]] = []

    for family in families:
        seeds = base_seeds + (hard_extra_seeds if family in HARD_OPERATOR_FAMILIES else [])
        for num_targets in range(min_num_targets, max_num_targets + 1):
            for synth_seed in seeds:
                bundle = build_generalized_synthetic_operator_bundle(
                    problem=problem,
                    family=family,
                    num_targets=num_targets,
                    seed=synth_seed,
                )
                try:
                    budget = validate_qprobe_request_budget(
                        target_names=bundle.target_names,
                        operator_map=bundle.operator_map,
                        has_custom_observables=True,
                    )
                except ValueError:
                    continue

                measurement_library = build_measurement_library_from_operator_map(bundle.operator_map)
                merged_groups = _merged_groups_for_targets(measurement_library, bundle.target_names)
                full_cost = len(merged_groups)
                zone = _zone(full_cost, exact_max_full_cost)

                bounded, bounded_meta = search_bounded_adaptive_plan_with_operator_map(
                    state=state,
                    operator_map=bundle.operator_map,
                    target_observables=bundle.target_names,
                    tolerance=tolerance,
                    shots_per_group=shots_per_group,
                    noise_model=noise_model,
                    seed=17,
                    node_budget=node_budget,
                )
                adaptive = search_adaptive_measurement_plan_with_operator_map(
                    state=state,
                    operator_map=bundle.operator_map,
                    target_observables=bundle.target_names,
                    tolerance=tolerance,
                    shots_per_group=shots_per_group,
                    noise_model=noise_model,
                    seed=17,
                )

                row: dict[str, object] = {
                    "family": problem.model_family,
                    "operator_family": family,
                    "num_targets": num_targets,
                    "full_cost": full_cost,
                    "subset_budget": int((2**full_cost) - 1),
                    "zone": zone,
                    "budget_num_targets": budget.num_targets,
                    "budget_total_pauli_terms": budget.total_pauli_terms,
                    "budget_max_operator_support": budget.max_operator_support,
                    "budget_basis_families": list(budget.basis_families),
                    "adaptive_cost": int(adaptive.final_plan.cost),
                    "adaptive_runtime_stop": bool(adaptive.success),
                    "adaptive_oracle_safe": bool(adaptive.oracle_benchmark_within_tolerance),
                    "bounded_cost": int(bounded.final_plan.cost),
                    "bounded_runtime_stop": bool(bounded.success),
                    "bounded_oracle_safe": bool(bounded.oracle_benchmark_within_tolerance),
                    "bounded_nodes_visited": bounded_meta.nodes_visited,
                    "bounded_lower_bound": int(bounded_meta.certified_lower_bound),
                    "bounded_gap_to_lb": int(bounded.final_plan.cost) - int(bounded_meta.certified_lower_bound),
                    "adaptive_gap_to_lb": int(adaptive.final_plan.cost) - int(bounded_meta.certified_lower_bound),
                }

                if zone != "frontier":
                    exact = search_minimal_measurement_plan_with_operator_map(
                        state=state,
                        operator_map=bundle.operator_map,
                        target_observables=bundle.target_names,
                        tolerance=tolerance,
                        shots_per_group=shots_per_group,
                        noise_model=noise_model,
                        seed=17,
                    )
                    exact_cost = int(exact.recommended_plan.cost)
                    row.update(
                        {
                            "exact_ran": True,
                            "exact_cost": exact_cost,
                            "adaptive_regret": int(adaptive.final_plan.cost) - exact_cost,
                            "bounded_regret": int(bounded.final_plan.cost) - exact_cost,
                        }
                    )
                else:
                    row.update(
                        {
                            "exact_ran": False,
                            "exact_cost": None,
                            "adaptive_regret": None,
                            "bounded_regret": None,
                        }
                    )

                rows.append(row)
    return rows


def _zone(full_cost: int, exact_max_full_cost: int) -> str:
    if full_cost <= 4:
        return "overlap"
    if full_cost <= exact_max_full_cost:
        return "edge"
    return "frontier"


def _exact_metrics(rows: list[dict[str, object]], regret_key: str, safe_key: str, stop_key: str) -> dict[str, object]:
    exact_rows = [row for row in rows if bool(row["exact_ran"])]
    if not exact_rows:
        return {"count": 0}
    regrets = [int(row[regret_key]) for row in exact_rows]
    count = len(exact_rows)
    regrets_sorted = sorted(regrets)
    median_regret = regrets_sorted[count // 2] if count % 2 == 1 else 0.5 * (
        regrets_sorted[count // 2 - 1] + regrets_sorted[count // 2]
    )
    return {
        "count": count,
        "avg_regret": sum(regrets) / count,
        "median_regret": median_regret,
        "within_optimal": sum(int(r <= 0) for r in regrets) / count,
        "within_plus_one": sum(int(r <= 1) for r in regrets) / count,
        "within_plus_two": sum(int(r <= 2) for r in regrets) / count,
        "oracle_safe_rate": sum(int(bool(row[safe_key])) for row in exact_rows) / count,
        "runtime_stop_rate": sum(int(bool(row[stop_key])) for row in exact_rows) / count,
        "avg_subset_budget": sum(int(row["subset_budget"]) for row in exact_rows) / count,
    }


def _frontier_metrics(rows: list[dict[str, object]], gap_key: str, safe_key: str, stop_key: str) -> dict[str, object]:
    frontier_rows = [row for row in rows if not bool(row["exact_ran"])]
    if not frontier_rows:
        return {"count": 0}
    gaps = [int(row[gap_key]) for row in frontier_rows]
    count = len(frontier_rows)
    gaps_sorted = sorted(gaps)
    median_gap = gaps_sorted[count // 2] if count % 2 == 1 else 0.5 * (
        gaps_sorted[count // 2 - 1] + gaps_sorted[count // 2]
    )
    return {
        "count": count,
        "avg_gap_to_lower_bound": sum(gaps) / count,
        "median_gap_to_lower_bound": median_gap,
        "within_lower_bound": sum(int(g <= 0) for g in gaps) / count,
        "within_lb_plus_one": sum(int(g <= 1) for g in gaps) / count,
        "within_lb_plus_two": sum(int(g <= 2) for g in gaps) / count,
        "oracle_safe_rate": sum(int(bool(row[safe_key])) for row in frontier_rows) / count,
        "runtime_stop_rate": sum(int(bool(row[stop_key])) for row in frontier_rows) / count,
        "avg_subset_budget": sum(int(row["subset_budget"]) for row in frontier_rows) / count,
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_zone: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_zone_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    full_cost_counts = Counter(int(row["full_cost"]) for row in rows)
    for row in rows:
        zone = str(row["zone"])
        fam = str(row["family"])
        by_zone[zone].append(row)
        by_family[fam].append(row)
        by_zone_family[f"{zone}|{fam}"].append(row)

    return {
        "total_requests": len(rows),
        "full_cost_distribution": dict(sorted(full_cost_counts.items())),
        "exact_overlap_edge": {
            "adaptive": _exact_metrics(rows, "adaptive_regret", "adaptive_oracle_safe", "adaptive_runtime_stop"),
            "bounded": _exact_metrics(rows, "bounded_regret", "bounded_oracle_safe", "bounded_runtime_stop"),
        },
        "frontier_certificate": {
            "adaptive": _frontier_metrics(rows, "adaptive_gap_to_lb", "adaptive_oracle_safe", "adaptive_runtime_stop"),
            "bounded": _frontier_metrics(rows, "bounded_gap_to_lb", "bounded_oracle_safe", "bounded_runtime_stop"),
        },
        "exact_by_zone": {
            zone: {
                "adaptive": _exact_metrics(items, "adaptive_regret", "adaptive_oracle_safe", "adaptive_runtime_stop"),
                "bounded": _exact_metrics(items, "bounded_regret", "bounded_oracle_safe", "bounded_runtime_stop"),
            }
            for zone, items in sorted(by_zone.items())
        },
        "frontier_by_zone": {
            zone: {
                "adaptive": _frontier_metrics(items, "adaptive_gap_to_lb", "adaptive_oracle_safe", "adaptive_runtime_stop"),
                "bounded": _frontier_metrics(items, "bounded_gap_to_lb", "bounded_oracle_safe", "bounded_runtime_stop"),
            }
            for zone, items in sorted(by_zone.items())
        },
        "exact_by_family": {
            fam: {
                "adaptive": _exact_metrics(items, "adaptive_regret", "adaptive_oracle_safe", "adaptive_runtime_stop"),
                "bounded": _exact_metrics(items, "bounded_regret", "bounded_oracle_safe", "bounded_runtime_stop"),
            }
            for fam, items in sorted(by_family.items())
        },
        "frontier_by_family": {
            fam: {
                "adaptive": _frontier_metrics(items, "adaptive_gap_to_lb", "adaptive_oracle_safe", "adaptive_runtime_stop"),
                "bounded": _frontier_metrics(items, "bounded_gap_to_lb", "bounded_oracle_safe", "bounded_runtime_stop"),
            }
            for fam, items in sorted(by_family.items())
        },
        "frontier_by_zone_family": {
            key: {
                "adaptive": _frontier_metrics(items, "adaptive_gap_to_lb", "adaptive_oracle_safe", "adaptive_runtime_stop"),
                "bounded": _frontier_metrics(items, "bounded_gap_to_lb", "bounded_oracle_safe", "bounded_runtime_stop"),
            }
            for key, items in sorted(by_zone_family.items())
        },
        "recommended_profile": _recommend_profile(by_zone),
    }


def _recommend_profile(by_zone: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    edge_metrics = _exact_metrics(by_zone.get("edge", []), "bounded_regret", "bounded_oracle_safe", "bounded_runtime_stop")
    frontier_metrics = _frontier_metrics(
        by_zone.get("frontier", []),
        "bounded_gap_to_lb",
        "bounded_oracle_safe",
        "bounded_runtime_stop",
    )
    return {
        "edge_ready": bool(edge_metrics.get("count")) and edge_metrics.get("within_plus_one", 0) >= 0.9,
        "frontier_certificate_ready": bool(frontier_metrics.get("count"))
        and frontier_metrics.get("within_lb_plus_one", 0) >= 0.9,
        "edge_metrics": edge_metrics,
        "frontier_metrics": frontier_metrics,
    }


def _ground_state(problem: ProblemSpec):
    if problem.model_family == "hubbard":
        h_op = build_hamiltonian(problem.Lx, problem.Ly, problem.t, problem.U, problem.mu)
    else:
        h_op = build_tfim_hamiltonian(problem.Lx, problem.Ly, problem.J, problem.h, problem.g)
    _, state = ground_state(h_op)
    return state


if __name__ == "__main__":
    main()

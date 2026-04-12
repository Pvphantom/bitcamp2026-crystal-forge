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
    search_adaptive_measurement_plan_with_operator_map,
    search_minimal_measurement_plan_with_operator_map,
)
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.tfim import build_tfim_hamiltonian


NOISE_LEVELS = [0.02, 0.08]
SHOT_COUNTS = [2000, 4000]
TOLERANCES = [0.01, 0.03]
NUM_TARGET_OPTIONS = [1, 2, 3]
BASE_SEEDS = [5, 11, 19]
HARD_EXTRA_SEEDS = [23, 29, 31, 37]

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
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "qprobe_adaptive_frontier_report.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--node-budget", type=int, default=64)
    parser.add_argument("--min-full-cost", type=int, default=1)
    parser.add_argument(
        "--family-scope",
        choices=("all", "hard"),
        default="all",
        help="Restrict benchmark generation to all generalized operator families or only the structurally hard ones.",
    )
    parser.add_argument(
        "--min-num-targets",
        type=int,
        default=1,
        help="Skip generated bundles with fewer than this many requested targets.",
    )
    args = parser.parse_args()

    report = benchmark_frontier(
        quick=args.quick,
        node_budget=args.node_budget,
        min_full_cost=args.min_full_cost,
        family_scope=args.family_scope,
        min_num_targets=args.min_num_targets,
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def benchmark_frontier(
    *,
    quick: bool,
    node_budget: int,
    min_full_cost: int,
    family_scope: str,
    min_num_targets: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    bundle_id = 0
    for point in HUBBARD_POINTS:
        problem = ProblemSpec.hubbard(**point)
        rows.extend(
            _benchmark_problem(
                problem,
                bundle_id_start=bundle_id + 1,
                quick=quick,
                node_budget=node_budget,
                min_full_cost=min_full_cost,
                family_scope=family_scope,
                min_num_targets=min_num_targets,
            )
        )
        bundle_id = len(rows)
    for point in TFIM_POINTS:
        problem = ProblemSpec.tfim(**point)
        rows.extend(
            _benchmark_problem(
                problem,
                bundle_id_start=bundle_id + 1,
                quick=quick,
                node_budget=node_budget,
                min_full_cost=min_full_cost,
                family_scope=family_scope,
                min_num_targets=min_num_targets,
            )
        )
        bundle_id = len(rows)
    report = _summarize(rows)
    report["node_budget"] = node_budget
    report["min_full_cost"] = min_full_cost
    report["family_scope"] = family_scope
    report["min_num_targets"] = min_num_targets
    return report


def _benchmark_problem(
    problem: ProblemSpec,
    *,
    bundle_id_start: int,
    quick: bool,
    node_budget: int,
    min_full_cost: int,
    family_scope: str,
    min_num_targets: int,
) -> list[dict[str, object]]:
    state = _ground_state(problem)
    rows: list[dict[str, object]] = []
    bundle_id = bundle_id_start
    noise_levels = [0.02] if quick else NOISE_LEVELS
    shot_counts = [2000] if quick else SHOT_COUNTS
    tolerances = [0.03] if quick else TOLERANCES
    num_target_options = [1, 2] if quick else NUM_TARGET_OPTIONS
    base_seeds = [5, 11] if quick else BASE_SEEDS
    hard_extra = [23, 29] if quick else HARD_EXTRA_SEEDS
    families = GENERALIZED_OPERATOR_FAMILIES if family_scope == "all" else tuple(sorted(HARD_OPERATOR_FAMILIES))

    for family in families:
        seeds = base_seeds + (hard_extra if family in HARD_OPERATOR_FAMILIES else [])
        for num_targets in num_target_options:
            if num_targets < min_num_targets:
                continue
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
                for readout_flip_prob in noise_levels:
                    for shots_per_group in shot_counts:
                        for tolerance in tolerances:
                            exact = search_minimal_measurement_plan_with_operator_map(
                                state=state,
                                operator_map=bundle.operator_map,
                                target_observables=bundle.target_names,
                                tolerance=tolerance,
                                shots_per_group=shots_per_group,
                                noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                                seed=17,
                            )
                            adaptive = search_adaptive_measurement_plan_with_operator_map(
                                state=state,
                                operator_map=bundle.operator_map,
                                target_observables=bundle.target_names,
                                tolerance=tolerance,
                                shots_per_group=shots_per_group,
                                noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                                seed=17,
                            )
                            bounded, bounded_meta = search_bounded_adaptive_plan_with_operator_map(
                                state=state,
                                operator_map=bundle.operator_map,
                                target_observables=bundle.target_names,
                                tolerance=tolerance,
                                shots_per_group=shots_per_group,
                                noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                                seed=17,
                                node_budget=node_budget,
                            )
                            full_cost = int(exact.full_plan.cost)
                            if full_cost < min_full_cost:
                                continue
                            exact_cost = int(exact.recommended_plan.cost)
                            adaptive_cost = int(adaptive.final_plan.cost)
                            adaptive_regret = adaptive_cost - exact_cost
                            bounded_cost = int(bounded.final_plan.cost)
                            bounded_regret = bounded_cost - exact_cost
                            rows.append(
                                {
                                    "id": bundle_id,
                                    "family": problem.model_family,
                                    "operator_family": family,
                                    "num_targets": num_targets,
                                    "readout_flip_prob": readout_flip_prob,
                                    "shots_per_group": shots_per_group,
                                    "tolerance": tolerance,
                                    "full_cost": full_cost,
                                    "exact_cost": exact_cost,
                                    "adaptive_cost": adaptive_cost,
                                    "adaptive_regret": adaptive_regret,
                                    "bounded_cost": bounded_cost,
                                    "bounded_regret": bounded_regret,
                                    "exact_success": bool(exact.success),
                                    "adaptive_runtime_stop": bool(adaptive.success),
                                    "adaptive_oracle_safe": bool(adaptive.oracle_benchmark_within_tolerance),
                                    "bounded_runtime_stop": bool(bounded.success),
                                    "bounded_oracle_safe": bool(bounded.oracle_benchmark_within_tolerance),
                                    "bounded_nodes_visited": bounded_meta.nodes_visited,
                                    "subset_budget": int((2**full_cost) - 1),
                                    "difficulty_band": _difficulty_band(full_cost),
                                    "budget_num_targets": budget.num_targets,
                                    "budget_total_pauli_terms": budget.total_pauli_terms,
                                    "budget_max_operator_support": budget.max_operator_support,
                                    "budget_basis_families": list(budget.basis_families),
                                }
                            )
                            bundle_id += 1
    return rows


def _difficulty_band(full_cost: int) -> str:
    if full_cost <= 4:
        return "easy"
    if full_cost <= 6:
        return "edge"
    return "frontier"


def _planner_metrics(rows: list[dict[str, object]], *, regret_key: str, safe_key: str, stop_key: str) -> dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "avg_regret": None,
            "median_regret": None,
            "within_optimal": None,
            "within_plus_one": None,
            "within_plus_two": None,
            "oracle_safe_rate": None,
            "runtime_stop_rate": None,
            "avg_subset_budget": None,
        }
    regrets = [int(row[regret_key]) for row in rows]
    count = len(rows)
    regrets_sorted = sorted(regrets)
    median_regret = regrets_sorted[count // 2] if count % 2 == 1 else 0.5 * (regrets_sorted[count // 2 - 1] + regrets_sorted[count // 2])
    return {
        "count": count,
        "avg_regret": sum(regrets) / count,
        "median_regret": median_regret,
        "within_optimal": sum(int(r <= 0) for r in regrets) / count,
        "within_plus_one": sum(int(r <= 1) for r in regrets) / count,
        "within_plus_two": sum(int(r <= 2) for r in regrets) / count,
        "oracle_safe_rate": sum(int(bool(row[safe_key])) for row in rows) / count,
        "runtime_stop_rate": sum(int(bool(row[stop_key])) for row in rows) / count,
        "avg_subset_budget": sum(int(row["subset_budget"]) for row in rows) / count,
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_band: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_band_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    full_cost_counts = Counter(int(row["full_cost"]) for row in rows)
    for row in rows:
        band = str(row["difficulty_band"])
        fam = str(row["family"])
        by_band[band].append(row)
        by_family[fam].append(row)
        by_band_family[f"{band}|{fam}"].append(row)

    return {
        "total_requests": len(rows),
        "full_cost_distribution": dict(sorted(full_cost_counts.items())),
        "adaptive_overall": _planner_metrics(rows, regret_key="adaptive_regret", safe_key="adaptive_oracle_safe", stop_key="adaptive_runtime_stop"),
        "bounded_overall": _planner_metrics(rows, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop"),
        "adaptive_by_band": {band: _planner_metrics(items, regret_key="adaptive_regret", safe_key="adaptive_oracle_safe", stop_key="adaptive_runtime_stop") for band, items in sorted(by_band.items())},
        "bounded_by_band": {band: _planner_metrics(items, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop") for band, items in sorted(by_band.items())},
        "adaptive_by_family": {fam: _planner_metrics(items, regret_key="adaptive_regret", safe_key="adaptive_oracle_safe", stop_key="adaptive_runtime_stop") for fam, items in sorted(by_family.items())},
        "bounded_by_family": {fam: _planner_metrics(items, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop") for fam, items in sorted(by_family.items())},
        "adaptive_by_band_family": {key: _planner_metrics(items, regret_key="adaptive_regret", safe_key="adaptive_oracle_safe", stop_key="adaptive_runtime_stop") for key, items in sorted(by_band_family.items())},
        "bounded_by_band_family": {key: _planner_metrics(items, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop") for key, items in sorted(by_band_family.items())},
        "recommended_frontier_profile": _recommend_profile(by_band),
    }


def _recommend_profile(by_band: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    for band in ("frontier", "edge", "easy"):
        rows = by_band.get(band, [])
        metrics = _planner_metrics(rows, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop")
        if metrics["count"] and metrics["within_plus_one"] is not None:
            if metrics["within_plus_one"] >= 0.9 and metrics["oracle_safe_rate"] >= 0.95:
                return {
                    "deploy_band": band,
                    "band_metrics": metrics,
                    "policy": "bounded_adaptive_qprobe_allowed",
                }
    return {
        "deploy_band": None,
        "policy": "bounded_adaptive_qprobe_not_yet_good_enough_beyond_exact_zone",
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

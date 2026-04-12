from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.analysis.qprobe_superconductor_channelize import build_superconductor_channel_plan
from app.analysis.qprobe_superconductor_decompose import (
    decompose_operator_map_by_transverse_signature,
)
from app.analysis.qprobe_superconductor_workflow import (
    build_superconductor_workflow_plan,
    bundle_orbit_map,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR
from app.observables.registry import build_default_observable_registry
from app.optimization.adaptive_bounded import search_bounded_adaptive_plan_with_operator_map
from app.optimization.measurement_plan import _merged_groups_for_targets
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.measurements import build_measurement_library_from_operator_map


HUBBARD_POINTS = [
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 1.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 6.0, "mu": 1.5},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 2.0},
]

SUPERCONDUCTOR_BUNDLES: dict[str, tuple[str, ...]] = {
    "charge_sector": ("n", "D"),
    "spin_sector": ("Ms2", "Cs_max"),
    "pair_sector": ("Pair_nn", "Pair_span"),
    "transport_pairing": ("K", "Pair_nn", "Pair_span"),
    "competing_orders": ("n", "Ms2", "Pair_nn", "Pair_span"),
    "superconductor_panel": ("D", "K", "Ms2", "Pair_nn", "Pair_span"),
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "qprobe_superconductor_report.json")
    parser.add_argument("--node-budget", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--shots-per-group", type=int, default=2000)
    parser.add_argument("--readout-flip-prob", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--heuristic-only", action="store_true")
    args = parser.parse_args()

    report = benchmark_superconductor(
        node_budget=args.node_budget,
        tolerance=args.tolerance,
        shots_per_group=args.shots_per_group,
        readout_flip_prob=args.readout_flip_prob,
        quick=args.quick,
        heuristic_only=args.heuristic_only,
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def benchmark_superconductor(
    *,
    node_budget: int,
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
    quick: bool = False,
    heuristic_only: bool = False,
) -> dict[str, object]:
    registry = build_default_observable_registry()
    rows: list[dict[str, object]] = []
    noise_model = NoiseModel(readout_flip_prob=readout_flip_prob)
    points = HUBBARD_POINTS[:2] if quick else HUBBARD_POINTS
    bundle_items = list(SUPERCONDUCTOR_BUNDLES.items())
    if quick:
        bundle_items = [item for item in bundle_items if item[0] in {"pair_sector", "transport_pairing", "competing_orders", "superconductor_panel"}]

    for point in points:
        problem = ProblemSpec.hubbard(**point)
        operator_map = {name: registry.operator(name, problem) for name in registry.names_for_family("hubbard")}
        _, state = ground_state(build_hamiltonian(problem.Lx, problem.Ly, problem.t, problem.U, problem.mu))
        for bundle_name, targets in bundle_items:
            submap = {name: operator_map[name] for name in targets}
            full_cost = len(_merged_groups_for_targets(build_measurement_library_from_operator_map(submap), targets))
            bounded, meta = search_bounded_adaptive_plan_with_operator_map(
                state=state,
                operator_map=submap,
                target_observables=targets,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                seed=17,
                node_budget=node_budget,
            )
            bounded_cost = int(bounded.final_plan.cost)
            plan = build_superconductor_channel_plan(operator_map=submap, targets=targets)
            channelized = _channelized_bounded_plan(
                state=state,
                operator_map=submap,
                plan=plan,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                node_budget=node_budget,
            )
            decomposed = _decomposed_bounded_plan(
                state=state,
                operator_map=submap,
                targets=targets,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                node_budget=node_budget,
            )
            workflow = _workflow_bounded_plan(
                state=state,
                operator_map=submap,
                targets=targets,
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                node_budget=node_budget,
            )
            rows.append(
                {
                    "bundle_name": bundle_name,
                    "targets": list(targets),
                    "U": problem.U,
                    "mu": problem.mu,
                    "full_cost": full_cost,
                    "subset_budget": int((2**full_cost) - 1),
                    "bounded_cost": bounded_cost,
                    "bounded_gap_to_lb": bounded_cost - int(meta.certified_lower_bound),
                    "bounded_oracle_safe": bool(bounded.oracle_benchmark_within_tolerance),
                    "bounded_runtime_stop": bool(bounded.success),
                    "bounded_nodes_visited": meta.nodes_visited,
                    "basis_concentration": plan.basis_concentration,
                    "mean_support": plan.mean_support,
                    "coherence_score": plan.coherence_score,
                    "num_channels": len(plan.channels),
                    "channelized_cost": channelized["cost"],
                    "channelized_gap_to_lb": channelized["gap_to_lb"],
                    "channelized_oracle_safe": channelized["oracle_safe"],
                    "channelized_runtime_stop": channelized["runtime_stop"],
                    "channelized_num_channels": channelized["num_channels"],
                    "channelized_safe_fraction": channelized["safe_fraction"],
                    "decomposed_cost": decomposed["cost"],
                    "decomposed_gap_to_lb": decomposed["gap_to_lb"],
                    "decomposed_oracle_safe": decomposed["oracle_safe"],
                    "decomposed_runtime_stop": decomposed["runtime_stop"],
                    "decomposed_num_targets": decomposed["num_targets"],
                    "decomposed_safe_fraction": decomposed["safe_fraction"],
                    "decomposed_avg_signature_groups_per_target": decomposed["avg_signature_groups_per_target"],
                    "workflow_cost": workflow["cost"],
                    "workflow_gap_to_lb": workflow["gap_to_lb"],
                    "workflow_oracle_safe": workflow["oracle_safe"],
                    "workflow_runtime_stop": workflow["runtime_stop"],
                    "workflow_safe_fraction": workflow["safe_fraction"],
                    "workflow_num_units": workflow["num_units"],
                    "workflow_num_direct_channels": workflow["num_direct_channels"],
                    "workflow_num_decomposed_channels": workflow["num_decomposed_channels"],
                    "workflow_recoverable_fraction": workflow["recoverable_fraction"],
                    "workflow_num_recoverable_units": workflow["num_recoverable_units"],
                }
            )
    report = _summarize(
        rows,
        node_budget=node_budget,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        readout_flip_prob=readout_flip_prob,
    )
    report["quick"] = quick
    report["heuristic_only"] = heuristic_only
    return report


def _channelized_bounded_plan(
    *,
    state,
    operator_map: dict[str, object],
    plan,
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel,
    node_budget: int,
) -> dict[str, object]:
    total_cost = 0
    total_lb = 0
    all_safe = True
    all_stop = True
    safe_count = 0
    for _, group_targets in plan.channels.items():
        subtargets = tuple(group_targets)
        submap = {name: operator_map[name] for name in subtargets}
        bounded, meta = search_bounded_adaptive_plan_with_operator_map(
            state=state,
            operator_map=submap,
            target_observables=subtargets,
            tolerance=tolerance,
            shots_per_group=shots_per_group,
            noise_model=noise_model,
            seed=17,
            node_budget=node_budget,
        )
        total_cost += int(bounded.final_plan.cost)
        total_lb += int(meta.certified_lower_bound)
        all_safe = all_safe and bool(bounded.oracle_benchmark_within_tolerance)
        all_stop = all_stop and bool(bounded.success)
        safe_count += int(bool(bounded.oracle_benchmark_within_tolerance))
    return {
        "cost": total_cost,
        "gap_to_lb": total_cost - total_lb,
        "oracle_safe": all_safe,
        "runtime_stop": all_stop,
        "num_channels": len(plan.channels),
        "safe_fraction": safe_count / max(len(plan.channels), 1),
    }


def _decomposed_bounded_plan(
    *,
    state,
    operator_map: dict[str, object],
    targets: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel,
    node_budget: int,
) -> dict[str, object]:
    decomposed = decompose_operator_map_by_transverse_signature(
        operator_map=operator_map,
        targets=targets,
    )
    total_cost = 0
    total_lb = 0
    all_safe = True
    all_stop = True
    safe_count = 0
    for target_name in decomposed.target_names:
        submap = {target_name: decomposed.operator_map[target_name]}
        bounded, meta = search_bounded_adaptive_plan_with_operator_map(
            state=state,
            operator_map=submap,
            target_observables=(target_name,),
            tolerance=tolerance,
            shots_per_group=shots_per_group,
            noise_model=noise_model,
            seed=17,
            node_budget=node_budget,
        )
        safe = bool(bounded.oracle_benchmark_within_tolerance)
        stop = bool(bounded.success)
        total_cost += int(bounded.final_plan.cost)
        total_lb += int(meta.certified_lower_bound)
        all_safe = all_safe and safe
        all_stop = all_stop and stop
        safe_count += int(safe)
    return {
        "cost": total_cost,
        "gap_to_lb": total_cost - total_lb,
        "oracle_safe": all_safe,
        "runtime_stop": all_stop,
        "num_targets": len(decomposed.target_names),
        "safe_fraction": safe_count / max(len(decomposed.target_names), 1),
        "avg_signature_groups_per_target": decomposed.avg_signature_groups_per_target,
    }


def _workflow_bounded_plan(
    *,
    state,
    operator_map: dict[str, object],
    targets: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    noise_model: NoiseModel,
    node_budget: int,
) -> dict[str, object]:
    workflow = build_superconductor_workflow_plan(
        operator_map=operator_map,
        targets=targets,
    )
    total_cost = 0
    total_lb = 0
    all_safe = True
    all_stop = True
    safe_count = 0
    num_units = 0
    recoverable_count = 0
    for channel_targets in workflow.direct_channels.values():
        for target_name in channel_targets:
            submap = {target_name: operator_map[target_name]}
            bounded, meta = search_bounded_adaptive_plan_with_operator_map(
                state=state,
                operator_map=submap,
                target_observables=(target_name,),
                tolerance=tolerance,
                shots_per_group=shots_per_group,
                noise_model=noise_model,
                seed=17,
                node_budget=node_budget,
            )
            safe = bool(bounded.oracle_benchmark_within_tolerance)
            stop = bool(bounded.success)
            total_cost += int(bounded.final_plan.cost)
            total_lb += int(meta.certified_lower_bound)
            all_safe = all_safe and safe
            all_stop = all_stop and stop
            safe_count += int(safe)
            recoverable_count += int(safe)
            num_units += 1
    for bundle in workflow.decomposed_channels.values():
        orbit_map = bundle_orbit_map(bundle)
        orbit_safe = 0
        for orbit_targets in orbit_map.values():
            orbit_has_safe = False
            for target_name in orbit_targets:
                submap = {target_name: bundle.operator_map[target_name]}
                bounded, meta = search_bounded_adaptive_plan_with_operator_map(
                    state=state,
                    operator_map=submap,
                    target_observables=(target_name,),
                    tolerance=tolerance,
                    shots_per_group=shots_per_group,
                    noise_model=noise_model,
                    seed=17,
                    node_budget=node_budget,
                )
                safe = bool(bounded.oracle_benchmark_within_tolerance)
                stop = bool(bounded.success)
                total_cost += int(bounded.final_plan.cost)
                total_lb += int(meta.certified_lower_bound)
                all_safe = all_safe and safe
                all_stop = all_stop and stop
                safe_count += int(safe)
                orbit_has_safe = orbit_has_safe or safe
                num_units += 1
            orbit_safe += int(orbit_has_safe)
            recoverable_count += int(orbit_has_safe)
        continue
    return {
        "cost": total_cost,
        "gap_to_lb": total_cost - total_lb,
        "oracle_safe": all_safe,
        "runtime_stop": all_stop,
        "safe_fraction": safe_count / max(num_units, 1),
        "recoverable_fraction": recoverable_count / max(
            len(tuple(t for ch in workflow.direct_channels.values() for t in ch)) + workflow.decomposed_orbit_count,
            1,
        ),
        "num_recoverable_units": recoverable_count,
        "num_units": num_units,
        "num_direct_channels": len(workflow.direct_channels),
        "num_decomposed_channels": len(workflow.decomposed_channels),
    }


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    return {
        "bounded": _planner_metrics(rows, cost_key="bounded_cost", gap_key="bounded_gap_to_lb", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop"),
        "channelized": _planner_metrics(rows, cost_key="channelized_cost", gap_key="channelized_gap_to_lb", safe_key="channelized_oracle_safe", stop_key="channelized_runtime_stop"),
        "decomposed": _planner_metrics(rows, cost_key="decomposed_cost", gap_key="decomposed_gap_to_lb", safe_key="decomposed_oracle_safe", stop_key="decomposed_runtime_stop"),
        "workflow": _planner_metrics(rows, cost_key="workflow_cost", gap_key="workflow_gap_to_lb", safe_key="workflow_oracle_safe", stop_key="workflow_runtime_stop"),
    }


def _planner_metrics(rows: list[dict[str, object]], *, cost_key: str, gap_key: str, safe_key: str, stop_key: str) -> dict[str, object]:
    count = len(rows)
    gaps = [int(row[gap_key]) for row in rows]
    gaps_sorted = sorted(gaps)
    median_gap = gaps_sorted[count // 2] if count % 2 == 1 else 0.5 * (
        gaps_sorted[count // 2 - 1] + gaps_sorted[count // 2]
    )
    return {
        "count": count,
        "avg_cost": sum(int(row[cost_key]) for row in rows) / count,
        "avg_gap_to_lb": sum(gaps) / count,
        "median_gap_to_lb": median_gap,
        "within_lower_bound": sum(int(g <= 0) for g in gaps) / count,
        "within_lb_plus_one": sum(int(g <= 1) for g in gaps) / count,
        "within_lb_plus_two": sum(int(g <= 2) for g in gaps) / count,
        "oracle_safe_rate": sum(int(bool(row[safe_key])) for row in rows) / count,
        "runtime_stop_rate": sum(int(bool(row[stop_key])) for row in rows) / count,
        "avg_subset_budget": sum(int(row["subset_budget"]) for row in rows) / count,
    }


def _summarize(
    rows: list[dict[str, object]],
    *,
    node_budget: int,
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
) -> dict[str, object]:
    by_bundle: dict[str, list[dict[str, object]]] = defaultdict(list)
    full_cost_counts = Counter(int(row["full_cost"]) for row in rows)
    for row in rows:
        by_bundle[str(row["bundle_name"])].append(row)
    return {
        "total_requests": len(rows),
        "full_cost_distribution": dict(sorted(full_cost_counts.items())),
        "overall": _metrics(rows),
        "by_bundle": {name: _metrics(items) for name, items in sorted(by_bundle.items())},
        "coherence_summary": {
            "avg_basis_concentration": sum(float(row["basis_concentration"]) for row in rows) / max(len(rows), 1),
            "avg_mean_support": sum(float(row["mean_support"]) for row in rows) / max(len(rows), 1),
            "avg_coherence_score": sum(float(row["coherence_score"]) for row in rows) / max(len(rows), 1),
            "avg_channelized_safe_fraction": sum(float(row["channelized_safe_fraction"]) for row in rows) / max(len(rows), 1),
            "avg_decomposed_safe_fraction": sum(float(row["decomposed_safe_fraction"]) for row in rows) / max(len(rows), 1),
            "avg_decomposed_signature_groups_per_target": sum(float(row["decomposed_avg_signature_groups_per_target"]) for row in rows) / max(len(rows), 1),
            "avg_workflow_safe_fraction": sum(float(row["workflow_safe_fraction"]) for row in rows) / max(len(rows), 1),
            "avg_workflow_num_units": sum(float(row["workflow_num_units"]) for row in rows) / max(len(rows), 1),
            "avg_workflow_recoverable_fraction": sum(float(row["workflow_recoverable_fraction"]) for row in rows) / max(len(rows), 1),
        },
        "node_budget": node_budget,
        "tolerance": tolerance,
        "shots_per_group": shots_per_group,
        "readout_flip_prob": readout_flip_prob,
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from app.analysis.general_tractability_features import analyze_general_tractability_features
from app.analysis.physics_first_corrmap import (
    PhysicsFirstCorrMapConfig,
    apply_physics_first_overlay,
    score_physics_first_corrmap,
)
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap
from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from scripts.eval_corrmap_on_regime_benchmark import _accuracy, _group_summary, _per_class_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_eval_physics_first.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_physics_first_corrmap(samples=samples)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_physics_first_corrmap(*, samples: list[dict]) -> dict[str, Any]:
    prepared = _prepare_samples(samples)
    benchmark_6x6 = [sample for sample in prepared if int(sample["problem"]["Lx"]) == 6]
    best_config = _select_config(benchmark_6x6)
    rows = _evaluate_rows(prepared, best_config)
    summary = _summarize(rows)
    summary["selected_config"] = best_config.__dict__
    summary["held_out_8x8_accuracy"] = _accuracy([row for row in rows if row["lattice"] == "8x8"])
    return {"summary": summary, "rows": rows}


def _select_config(samples: list[dict]) -> PhysicsFirstCorrMapConfig:
    best_config = PhysicsFirstCorrMapConfig()
    best_score = float("-inf")
    mean_thresholds = [0.6, 0.66]
    quantum_thresholds = [0.6, 0.66]
    margins = [0.0, 0.05]
    weights = [0.95, 1.05]
    for mw, sw, qw, mt, qt, mm, qm in itertools.product(weights, weights, weights, mean_thresholds, quantum_thresholds, margins, margins):
        config = PhysicsFirstCorrMapConfig(
            mean_field_weight=mw,
            scalable_weight=sw,
            quantum_weight=qw,
            mean_field_threshold=mt,
            quantum_threshold=qt,
            mean_field_margin=mm,
            quantum_margin=qm,
        )
        rows = _evaluate_rows(samples, config)
        summary = _summarize(rows)
        score = (
            float(summary["overall_accuracy"])
            + 0.25 * float(summary["per_class_metrics"].get("mean_field", {}).get("f1", 0.0))
            + 0.25 * float(summary["per_class_metrics"].get("scalable_classical", {}).get("f1", 0.0))
            + 0.15 * float(summary["per_class_metrics"].get("quantum_frontier", {}).get("f1", 0.0))
            + 0.35 * float(summary["by_family"].get("tfim", {}).get("accuracy", 0.0))
        )
        if score > best_score:
            best_score = score
            best_config = config
    return best_config


def _prepare_samples(samples: list[dict]) -> list[dict[str, Any]]:
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()
    prepared: list[dict[str, Any]] = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
        general = analyze_general_tractability_features(
            cheap_result=cheap,
            stability=runtime.stability,
            sensitivity=runtime.sensitivity,
            size_consistency=runtime.size_consistency,
            ansatz_disagreement=runtime.ansatz_disagreement,
            hysteresis=runtime.hysteresis,
            physical_tractability=runtime.physical_tractability,
        )
        prepared.append(
            {
                **sample,
                "_problem_spec": problem,
                "_runtime": runtime,
                "_general": general,
            }
        )
    return prepared


def _evaluate_rows(samples: list[dict], config: PhysicsFirstCorrMapConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        problem = sample["_problem_spec"]
        runtime = sample["_runtime"]
        general = sample["_general"]
        physics = score_physics_first_corrmap(runtime=runtime, general=general, config=config)
        final = apply_physics_first_overlay(base_prediction=None, physics_report=physics, runtime=runtime)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "benchmark_label": sample["benchmark_label"],
                "predicted_label": final["label"],
                "correct": final["label"] == sample["benchmark_label"],
                "model_family": problem.model_family,
                "lattice": f"{problem.Lx}x{problem.Ly}",
                "parameters": dict(problem.parameters.values),
                "rationale": sample["rationale"],
                "intrinsic_label": final.get("intrinsic_label"),
                "intrinsic_score": final.get("intrinsic_score"),
                "confidence": final.get("confidence"),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_truth[row["benchmark_label"]].append(row)
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
    predicted_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        predicted_counts[row["predicted_label"]] += 1
    return {
        "overall_accuracy": _accuracy(rows),
        "truth_label_counts": dict((label, len(group)) for label, group in by_truth.items()),
        "predicted_label_counts": dict(predicted_counts),
        "per_class_metrics": _per_class_metrics(rows),
        "by_truth_label": {name: _group_summary(group) for name, group in sorted(by_truth.items())},
        "by_family": {name: _group_summary(group) for name, group in sorted(by_family.items())},
        "by_lattice": {name: _group_summary(group) for name, group in sorted(by_lattice.items())},
    }


def _problem_from_payload(payload: dict[str, Any]) -> ProblemSpec:
    family = str(payload["model_family"])
    params = dict(payload["parameters"])
    if family == "hubbard":
        return ProblemSpec.hubbard(
            Lx=int(payload["Lx"]),
            Ly=int(payload["Ly"]),
            t=float(params["t"]),
            U=float(params["U"]),
            mu=float(params["mu"]),
            boundary=str(payload.get("boundary", "open")),
        )
    return ProblemSpec.tfim(
        Lx=int(payload["Lx"]),
        Ly=int(payload["Ly"]),
        J=float(params["J"]),
        h=float(params["h"]),
        g=float(params.get("g", 0.0)),
        boundary=str(payload.get("boundary", "open")),
    )


if __name__ == "__main__":
    main()

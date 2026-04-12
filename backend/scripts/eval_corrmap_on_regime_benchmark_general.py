from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from app.analysis.general_tractability_features import analyze_general_tractability_features
from app.analysis.intrinsic_feature_vector_general import build_runtime_general_features
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap, apply_runtime_intrinsic_overlay
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.infer import HybridCorrMapInferenceEngine
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from scripts.eval_corrmap_on_regime_benchmark import _group_summary, _per_class_metrics, _accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/hybrid_corrmap_general.pt"))
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_eval_general.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_regime_benchmark_general(samples=samples, model_path=args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_regime_benchmark_general(*, samples: list[dict], model_path: Path) -> dict[str, Any]:
    inference = HybridCorrMapInferenceEngine(model_path=model_path)
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()

    rows: list[dict[str, Any]] = []
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
        base_features = build_trust_feature_vector(problem, cheap)
        features = build_runtime_general_features(base_features, runtime, general)
        prediction = inference.predict(features)
        final = apply_runtime_intrinsic_overlay(prediction, runtime)
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
                "abstain_reason": final.get("abstain_reason"),
                "confidence": final.get("confidence"),
            }
        )

    by_truth: dict[str, list[dict[str, Any]]] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    by_lattice: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_truth.setdefault(row["benchmark_label"], []).append(row)
        by_family.setdefault(row["model_family"], []).append(row)
        by_lattice.setdefault(row["lattice"], []).append(row)

    return {
        "summary": {
            "overall_accuracy": _accuracy(rows),
            "truth_label_counts": {label: len(group) for label, group in by_truth.items()},
            "predicted_label_counts": _count_predicted(rows),
            "per_class_metrics": _per_class_metrics(rows),
            "by_truth_label": {name: _group_summary(group) for name, group in sorted(by_truth.items())},
            "by_family": {name: _group_summary(group) for name, group in sorted(by_family.items())},
            "by_lattice": {name: _group_summary(group) for name, group in sorted(by_lattice.items())},
        },
        "rows": rows,
    }


def _count_predicted(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["predicted_label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


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
    if family == "tfim":
        return ProblemSpec.tfim(
            Lx=int(payload["Lx"]),
            Ly=int(payload["Ly"]),
            J=float(params["J"]),
            h=float(params["h"]),
            g=float(params.get("g", 0.0)),
            boundary=str(payload.get("boundary", "open")),
        )
    raise ValueError(f"Unsupported model family: {family}")


if __name__ == "__main__":
    main()

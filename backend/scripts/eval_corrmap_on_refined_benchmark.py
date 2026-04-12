from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from app.analysis.intrinsic_feature_vector import build_runtime_augmented_features
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap, apply_runtime_intrinsic_overlay
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.infer import HybridCorrMapInferenceEngine
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


LABELS = ["mean_field", "scalable_classical", "quantum_frontier"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark_refined.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/hybrid_corrmap_augmented.pt"))
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_refined_eval.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_refined_benchmark(samples=samples, model_path=args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_refined_benchmark(*, samples: list[dict], model_path: Path) -> dict[str, Any]:
    inference = HybridCorrMapInferenceEngine(model_path=model_path)
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
        base_features = build_trust_feature_vector(problem, cheap)
        features = build_runtime_augmented_features(base_features, runtime)
        prediction = inference.predict(features)
        final = apply_runtime_intrinsic_overlay(prediction, runtime)
        pred = str(final["label"])
        probs = dict(sample["benchmark_label_probs"])
        hard_correct = pred == sample["benchmark_label"]
        soft_score = float(probs.get(pred, 0.0))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "benchmark_label": sample["benchmark_label"],
                "benchmark_label_probs": probs,
                "benchmark_confidence": sample["benchmark_confidence"],
                "predicted_label": pred,
                "hard_correct": hard_correct,
                "soft_score": soft_score,
                "model_family": problem.model_family,
                "lattice": f"{problem.Lx}x{problem.Ly}",
                "parameters": dict(problem.parameters.values),
                "intrinsic_label": final.get("intrinsic_label"),
                "intrinsic_score": final.get("intrinsic_score"),
            }
        )

    return {"summary": _summarize(rows), "rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_conf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
        by_conf[row["benchmark_confidence"]].append(row)
    return {
        "hard_accuracy": _mean(row["hard_correct"] for row in rows),
        "soft_accuracy": _mean(row["soft_score"] for row in rows),
        "by_family": {k: _group_summary(v) for k, v in sorted(by_family.items())},
        "by_lattice": {k: _group_summary(v) for k, v in sorted(by_lattice.items())},
        "by_confidence": {k: _group_summary(v) for k, v in sorted(by_conf.items())},
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_samples": float(len(rows)),
        "hard_accuracy": _mean(row["hard_correct"] for row in rows),
        "soft_accuracy": _mean(row["soft_score"] for row in rows),
    }


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return 0.0 if not vals else sum(vals) / len(vals)


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

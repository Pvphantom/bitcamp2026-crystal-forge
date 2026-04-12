from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from app.analysis.intrinsic_feature_vector import build_runtime_augmented_features
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap, apply_runtime_intrinsic_overlay
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.infer import HybridCorrMapInferenceEngine
from app.ml.structured_corrmap_model import StructuredCorrMapMLP
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("backend/artifacts/regime_benchmark.pt"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("backend/artifacts/hybrid_corrmap_test.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/artifacts/regime_benchmark_eval.json"),
    )
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_regime_benchmark(samples=samples, model_path=args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_regime_benchmark(*, samples: list[dict], model_path: Path) -> dict[str, Any]:
    predictor = _build_predictor(model_path)
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()

    rows: list[dict[str, Any]] = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
        features = build_trust_feature_vector(problem, cheap)
        prediction = predictor(problem, features, runtime)
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

    return {
        "summary": _summarize(rows),
        "rows": rows,
    }


def _build_predictor(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if checkpoint.get("model_type") == "structured_family":
        model = StructuredCorrMapMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        feature_mean = checkpoint["feature_mean"]
        feature_std = checkpoint["feature_std"].clamp_min(1e-6)

        def predict(problem: ProblemSpec, features: torch.Tensor | Any, runtime) -> dict[str, Any]:
            feature_tensor = build_runtime_augmented_features(torch.as_tensor(features, dtype=torch.float32), runtime)
            normalized = (feature_tensor - feature_mean) / feature_std
            model.eval()
            with torch.no_grad():
                outputs = model(normalized.unsqueeze(0))
                mean_field_prob = float(torch.sigmoid(outputs["mean_field_logit"])[0].item())
                quantum_prob = float(torch.sigmoid(outputs["quantum_frontier_logit"])[0].item())
            if quantum_prob >= 0.5:
                label = "quantum_frontier"
            elif mean_field_prob >= 0.7:
                label = "mean_field"
            else:
                label = "scalable_classical"
            middle = max(0.0, 1.0 - max(mean_field_prob, quantum_prob))
            return {
                "label": label,
                "recommended_action": label,
                "candidate_scores": {
                    "mean_field": mean_field_prob,
                    "quantum_frontier": quantum_prob,
                    "scalable_classical": middle,
                },
                "abstained": False,
                "abstain_reason": None,
                "confidence": max(mean_field_prob, quantum_prob, middle),
            }

        return predict

    inference = HybridCorrMapInferenceEngine(model_path=model_path)

    def predict(problem: ProblemSpec, features: torch.Tensor | Any, runtime) -> dict[str, Any] | None:
        augmented = build_runtime_augmented_features(torch.as_tensor(features, dtype=torch.float32), runtime)
        return inference.predict(augmented)

    return predict


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_truth[row["benchmark_label"]].append(row)
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
    return {
        "overall_accuracy": _accuracy(rows),
        "truth_label_counts": dict(Counter(row["benchmark_label"] for row in rows)),
        "predicted_label_counts": dict(Counter(row["predicted_label"] for row in rows)),
        "per_class_metrics": _per_class_metrics(rows),
        "by_truth_label": {name: _group_summary(group) for name, group in sorted(by_truth.items())},
        "by_family": {name: _group_summary(group) for name, group in sorted(by_family.items())},
        "by_lattice": {name: _group_summary(group) for name, group in sorted(by_lattice.items())},
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(rows),
        "accuracy": _accuracy(rows),
        "predicted_label_counts": dict(Counter(row["predicted_label"] for row in rows)),
        "intrinsic_label_counts": dict(Counter(str(row["intrinsic_label"]) for row in rows)),
    }


def _per_class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    labels = sorted({row["benchmark_label"] for row in rows} | {row["predicted_label"] for row in rows})
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for row in rows if row["benchmark_label"] == label and row["predicted_label"] == label)
        fp = sum(1 for row in rows if row["benchmark_label"] != label and row["predicted_label"] == label)
        fn = sum(1 for row in rows if row["benchmark_label"] == label and row["predicted_label"] != label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["correct"]) / len(rows)


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

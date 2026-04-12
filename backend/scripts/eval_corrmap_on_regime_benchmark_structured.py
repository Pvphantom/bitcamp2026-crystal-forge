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
from app.ml.structured_corrmap_model import StructuredCorrMapMLP
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from scripts.eval_corrmap_on_regime_benchmark import _accuracy, _group_summary, _per_class_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_eval_structured.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_regime_benchmark_structured(samples=samples, model_path=args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_regime_benchmark_structured(*, samples: list[dict], model_path: Path) -> dict[str, Any]:
    predictor = StructuredCorrMapPredictor(model_path)
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
        base_features = build_trust_feature_vector(problem, cheap)
        features = build_runtime_augmented_features(base_features, runtime)
        prediction = predictor.predict(features)
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
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_truth[row["benchmark_label"]].append(row)
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
    return {
        "summary": {
            "overall_accuracy": _accuracy(rows),
            "truth_label_counts": dict((name, len(group)) for name, group in by_truth.items()),
            "predicted_label_counts": dict((name, sum(1 for row in rows if row["predicted_label"] == name)) for name in {row["predicted_label"] for row in rows}),
            "per_class_metrics": _per_class_metrics(rows),
            "by_truth_label": {name: _group_summary(group) for name, group in sorted(by_truth.items())},
            "by_family": {name: _group_summary(group) for name, group in sorted(by_family.items())},
            "by_lattice": {name: _group_summary(group) for name, group in sorted(by_lattice.items())},
        },
        "rows": rows,
    }


class StructuredCorrMapPredictor:
    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        self._loaded = False
        self._model: StructuredCorrMapMLP | None = None
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None
        self._mean_field_threshold: float = 0.75
        self._quantum_threshold: float = 0.5

    def predict(self, features: torch.Tensor | Any) -> dict[str, Any]:
        model = self._load()
        assert self._feature_mean is not None and self._feature_std is not None
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = (feature_tensor - self._feature_mean) / self._feature_std
        model.eval()
        with torch.no_grad():
            outputs = model(normalized.unsqueeze(0))
            mean_field_prob = float(torch.sigmoid(outputs["mean_field_logit"])[0].item())
            quantum_prob = float(torch.sigmoid(outputs["quantum_frontier_logit"])[0].item())
            intrinsic_probs = torch.softmax(outputs["intrinsic_logits"], dim=-1)[0]
        if quantum_prob >= self._quantum_threshold:
            label = "quantum_frontier"
        elif mean_field_prob >= self._mean_field_threshold:
            label = "mean_field"
        else:
            label = "scalable_classical"
        middle = max(0.0, 1.0 - max(mean_field_prob, quantum_prob))
        intrinsic_labels = ["stable_classical", "fragile_classical", "frontier_or_uncertain"]
        return {
            "available": True,
            "model_path": str(self.model_path),
            "label": label,
            "confidence": max(mean_field_prob, quantum_prob, middle),
            "recommended_action": label,
            "candidate_scores": {
                "mean_field": mean_field_prob,
                "quantum_frontier": quantum_prob,
                "scalable_classical": middle,
            },
            "abstained": False,
            "abstain_reason": None,
            "intrinsic_label": intrinsic_labels[int(torch.argmax(intrinsic_probs).item())],
            "intrinsic_score": float(torch.max(intrinsic_probs).item()),
        }

    def _load(self) -> StructuredCorrMapMLP:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = StructuredCorrMapMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"].clamp_min(1e-6)
        self._mean_field_threshold = float(checkpoint.get("mean_field_threshold", 0.75))
        self._quantum_threshold = float(checkpoint.get("quantum_threshold", 0.5))
        self._model = model
        self._loaded = True
        return model


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

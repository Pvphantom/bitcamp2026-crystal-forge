from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from app.analysis.intrinsic_feature_vector import build_runtime_augmented_features
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.binary_corrmap_model import BinaryCorrMapMLP
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


POSITIVE_LABEL = "quantum_frontier"
NEGATIVE_LABEL = "classical_scalable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark_refined_v2.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/binary_corrmap.pt"))
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_binary_eval.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_binary_regime_benchmark(samples=samples, model_path=args.model)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def _binary_label(label: str) -> str:
    return POSITIVE_LABEL if label == POSITIVE_LABEL else NEGATIVE_LABEL


def evaluate_binary_regime_benchmark(*, samples: list[dict], model_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = BinaryCorrMapMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"].clamp_min(1e-6)
    threshold = float(checkpoint["threshold"])

    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()

    rows: list[dict[str, Any]] = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
        base = build_trust_feature_vector(problem, cheap)
        features = build_runtime_augmented_features(base, runtime)
        normalized = (features - feature_mean) / feature_std
        with torch.no_grad():
            prob = float(torch.sigmoid(model(normalized.unsqueeze(0)))[0].item())
        pred = POSITIVE_LABEL if prob >= threshold else NEGATIVE_LABEL
        truth = _binary_label(sample["benchmark_label"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "benchmark_label": truth,
                "source_label": sample["benchmark_label"],
                "predicted_label": pred,
                "correct": pred == truth,
                "prob_quantum_frontier": prob,
                "model_family": problem.model_family,
                "lattice": f"{problem.Lx}x{problem.Ly}",
                "parameters": dict(problem.parameters.values),
            }
        )

    return {"summary": _summarize(rows), "rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
        by_truth[row["benchmark_label"]].append(row)
    return {
        "overall_accuracy": _accuracy(rows),
        "truth_label_counts": dict(Counter(row["benchmark_label"] for row in rows)),
        "predicted_label_counts": dict(Counter(row["predicted_label"] for row in rows)),
        "by_family": {k: _group_summary(v) for k, v in sorted(by_family.items())},
        "by_lattice": {k: _group_summary(v) for k, v in sorted(by_lattice.items())},
        "by_truth_label": {k: _group_summary(v) for k, v in sorted(by_truth.items())},
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"num_samples": len(rows), "accuracy": _accuracy(rows), "predicted_label_counts": dict(Counter(row["predicted_label"] for row in rows))}


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return 0.0 if not rows else sum(1 for row in rows if row["correct"]) / len(rows)


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

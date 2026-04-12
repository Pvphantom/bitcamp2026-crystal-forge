from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from app.analysis.runtime_intrinsic_corrmap import (
    analyze_runtime_intrinsic_corrmap,
    apply_runtime_intrinsic_overlay,
)
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.infer import HybridCorrMapInferenceEngine, RoutingInferenceEngine
from app.ml.schema import DEFAULT_HYBRID_CORRMAP_MODEL_PATH, DEFAULT_ROUTING_DATASET
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from scripts.train_routing_model import train


FRONTIER_GRIDS = {
    ("hubbard", 4, 4): {
        "U": [1.0, 2.0, 4.0, 6.0, 8.0],
        "mu": [0.0, 2.0, 4.0],
    },
    ("hubbard", 6, 6): {
        "U": [1.0, 4.0, 8.0],
        "mu": [0.0, 2.0, 4.0],
    },
    ("tfim", 4, 4): {
        "J": [0.5, 1.0, 1.5],
        "h": [0.4, 1.0, 1.8],
        "g": [0.0, 0.5],
    },
    ("tfim", 6, 6): {
        "J": [1.0],
        "h": [0.4, 1.0, 1.8],
        "g": [0.0, 0.5],
    },
}


def filter_small_lattice_training_samples(samples: list[dict], *, max_train_nsites: int) -> list[dict]:
    filtered = []
    for sample in samples:
        nsites = int(sample.get("problem_metadata", {}).get("nsites", 0))
        if nsites <= max_train_nsites:
            filtered.append(sample)
    return filtered


def build_frontier_prediction_rows() -> list[dict[str, Any]]:
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for (family, Lx, Ly), grid in FRONTIER_GRIDS.items():
        for problem in _problems_for_grid(family, Lx, Ly, grid):
            sample_id += 1
            cheap_solver = mean_field if family == "hubbard" else tfim_mean_field
            cheap_result = cheap_solver.solve(problem)
            features = build_trust_feature_vector(problem, cheap_result)
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_family": family,
                    "lattice": f"{Lx}x{Ly}",
                    "nsites": problem.nsites,
                    "parameters": dict(problem.parameters.values),
                    "cheap_solver": cheap_result.solver_name,
                    "cheap_energy": float(cheap_result.energy),
                    "cheap_observables": {key: float(value) for key, value in cheap_result.global_observables.items()},
                    "features": features,
                    "problem": problem,
                }
            )
    return rows


def run_frontier_predictions(rows: list[dict[str, Any]], *, inference) -> dict[str, Any]:
    rendered_rows = []
    for row in rows:
        prediction = inference.predict(row["features"])
        runtime_intrinsic = analyze_runtime_intrinsic_corrmap(row["problem"])
        prediction = apply_runtime_intrinsic_overlay(prediction, runtime_intrinsic)
        rendered_rows.append(
            {
                **_serializable_row(row),
                "route_label": prediction["label"],
                "abstained": bool(prediction.get("abstained", False)),
                "abstain_reason": prediction.get("abstain_reason"),
                "confidence": prediction.get("confidence"),
                "candidate_scores": prediction.get("candidate_scores", {}),
                "intrinsic_label": prediction.get("intrinsic_label"),
                "intrinsic_score": prediction.get("intrinsic_score"),
                "intrinsic_reasons": prediction.get("intrinsic_reasons", []),
            }
        )
    return {
        "summary": summarize_frontier_predictions(rendered_rows),
        "rows": rendered_rows,
    }


def summarize_frontier_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_lattice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_intrinsic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["model_family"]].append(row)
        by_lattice[row["lattice"]].append(row)
        intrinsic_label = row.get("intrinsic_label")
        if intrinsic_label is not None:
            by_intrinsic[str(intrinsic_label)].append(row)
    return {
        "overall": _frontier_group_summary(rows),
        "by_family": {name: _frontier_group_summary(group) for name, group in sorted(by_family.items())},
        "by_lattice": {name: _frontier_group_summary(group) for name, group in sorted(by_lattice.items())},
        "by_intrinsic_label": {name: _frontier_group_summary(group) for name, group in sorted(by_intrinsic.items())},
    }


def _frontier_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "num_samples": 0,
            "route_label_counts": {},
            "abstention_rate": None,
            "mean_confidence": None,
            "median_confidence": None,
            "intrinsic_label_counts": {},
            "intrinsic_guard_rate": None,
        }
    confidences = [float(row["confidence"]) for row in rows if row["confidence"] is not None]
    abstentions = [row for row in rows if row["abstained"]]
    intrinsic_guard_rows = [row for row in abstentions if row["abstain_reason"] == "intrinsic_risk_guard"]
    return {
        "num_samples": total,
        "route_label_counts": dict(Counter(row["route_label"] for row in rows)),
        "abstention_rate": len(abstentions) / total,
        "mean_confidence": None if not confidences else float(sum(confidences) / len(confidences)),
        "median_confidence": None if not confidences else float(sorted(confidences)[len(confidences) // 2]),
        "intrinsic_label_counts": dict(Counter(row["intrinsic_label"] for row in rows if row.get("intrinsic_label") is not None)),
        "intrinsic_guard_rate": len(intrinsic_guard_rows) / total,
    }


def orchestrate_frontier_benchmark(
    *,
    train_dataset_path: Path,
    model_out: Path,
    train_metrics_out: Path,
    benchmark_report_out: Path,
    max_train_nsites: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_hybrid_model: bool = False,
    hybrid_model_path: Path | None = None,
) -> dict[str, Any]:
    raw_train_samples = torch.load(train_dataset_path, map_location="cpu")
    train_samples = filter_small_lattice_training_samples(raw_train_samples, max_train_nsites=max_train_nsites)
    if not train_samples:
        raise ValueError("No training samples remain after small-lattice filtering")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        filtered_dataset_path = Path(handle.name)
    try:
        torch.save(train_samples, filtered_dataset_path)
        train_metrics = train(
            dataset_path=filtered_dataset_path,
            model_out=model_out,
            metrics_out=train_metrics_out,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            allow_weak_labels=False,
            include_uncertain=False,
        )
    finally:
        filtered_dataset_path.unlink(missing_ok=True)

    if use_hybrid_model:
        inference = HybridCorrMapInferenceEngine(model_path=hybrid_model_path or DEFAULT_HYBRID_CORRMAP_MODEL_PATH)
    else:
        inference = RoutingInferenceEngine(model_path=model_out)
    frontier_rows = build_frontier_prediction_rows()
    frontier = run_frontier_predictions(frontier_rows, inference=inference)

    report = {
        "training_policy": {
            "max_train_nsites": max_train_nsites,
            "allowed_reference_qualities": ["strong"],
            "include_uncertain_labels": False,
            "large_lattice_training_labels_used": 0,
            "num_raw_training_samples": len(raw_train_samples),
            "num_filtered_training_samples": len(train_samples),
            "filtered_out_large_training_samples": len(raw_train_samples) - len(train_samples),
            "max_training_nsites_observed": max(
                int(sample.get("problem_metadata", {}).get("nsites", 0))
                for sample in train_samples
            ),
            "inference_mode": "hybrid" if use_hybrid_model else "routing_only",
        },
        "training_metrics": train_metrics,
        "frontier_predictions": frontier,
    }
    benchmark_report_out.write_text(json.dumps(report, indent=2))
    return report


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "model_family": row["model_family"],
        "lattice": row["lattice"],
        "nsites": row["nsites"],
        "parameters": row["parameters"],
        "cheap_solver": row["cheap_solver"],
        "cheap_energy": row["cheap_energy"],
        "cheap_observables": row["cheap_observables"],
    }


def _problems_for_grid(family: str, Lx: int, Ly: int, grid: dict[str, list[float]]) -> list[ProblemSpec]:
    problems: list[ProblemSpec] = []
    if family == "hubbard":
        for U in grid["U"]:
            for mu in grid["mu"]:
                problems.append(ProblemSpec.hubbard(Lx=Lx, Ly=Ly, t=1.0, U=U, mu=mu))
        return problems
    if family == "tfim":
        for J in grid["J"]:
            for h in grid["h"]:
                for g in grid["g"]:
                    problems.append(ProblemSpec.tfim(Lx=Lx, Ly=Ly, J=J, h=h, g=g))
        return problems
    raise ValueError(f"Unsupported model family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_ROUTING_DATASET)
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/frontier_routing_model.pt"))
    parser.add_argument("--train-metrics-out", type=Path, default=Path("backend/artifacts/frontier_routing_metrics.json"))
    parser.add_argument("--benchmark-report-out", type=Path, default=Path("backend/artifacts/frontier_benchmark_report.json"))
    parser.add_argument("--max-train-nsites", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--use-hybrid-model", action="store_true")
    parser.add_argument("--hybrid-model-path", type=Path, default=DEFAULT_HYBRID_CORRMAP_MODEL_PATH)
    args = parser.parse_args()

    report = orchestrate_frontier_benchmark(
        train_dataset_path=args.train_dataset,
        model_out=args.model_out,
        train_metrics_out=args.train_metrics_out,
        benchmark_report_out=args.benchmark_report_out,
        max_train_nsites=args.max_train_nsites,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_hybrid_model=args.use_hybrid_model,
        hybrid_model_path=args.hybrid_model_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

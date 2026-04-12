from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap, apply_runtime_intrinsic_overlay
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.infer import HybridCorrMapInferenceEngine
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from scripts.eval_corrmap_on_regime_benchmark import _summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--hubbard-model", type=Path, required=True)
    parser.add_argument("--tfim-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/regime_benchmark_eval_family_models.json"))
    args = parser.parse_args()

    samples = torch.load(args.benchmark, map_location="cpu")
    report = evaluate_regime_benchmark_family_models(
        samples=samples,
        hubbard_model_path=args.hubbard_model,
        tfim_model_path=args.tfim_model,
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


def evaluate_regime_benchmark_family_models(
    *,
    samples: list[dict],
    hubbard_model_path: Path,
    tfim_model_path: Path,
) -> dict:
    engines = {
        "hubbard": HybridCorrMapInferenceEngine(model_path=hubbard_model_path),
        "tfim": HybridCorrMapInferenceEngine(model_path=tfim_model_path),
    }
    mean_field = MeanFieldSolver()
    tfim_mean_field = TFIMMeanFieldSolver()
    rows = []
    for sample in samples:
        problem = _problem_from_payload(sample["problem"])
        cheap = mean_field.solve(problem) if problem.model_family == "hubbard" else tfim_mean_field.solve(problem)
        features = build_trust_feature_vector(problem, cheap)
        prediction = engines[problem.model_family].predict(features)
        runtime = analyze_runtime_intrinsic_corrmap(problem)
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
    return {"summary": _summarize(rows), "rows": rows}


def _problem_from_payload(payload: dict) -> ProblemSpec:
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

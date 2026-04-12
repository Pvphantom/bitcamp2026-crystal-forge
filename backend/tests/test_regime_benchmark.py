from pathlib import Path

import torch

from scripts.build_regime_benchmark import build_regime_benchmark
from scripts.eval_corrmap_on_regime_benchmark import evaluate_regime_benchmark


def test_regime_benchmark_has_balanced_labels() -> None:
    samples = build_regime_benchmark()
    counts = {}
    for sample in samples:
        counts[sample["benchmark_label"]] = counts.get(sample["benchmark_label"], 0) + 1
    assert len(samples) == 60
    assert counts == {
        "mean_field": 20,
        "scalable_classical": 20,
        "quantum_frontier": 20,
    }


def test_regime_benchmark_eval_runs_with_current_hybrid_model() -> None:
    model_path = Path("backend/artifacts/hybrid_corrmap_test.pt")
    if not model_path.exists():
        return
    samples = build_regime_benchmark()
    report = evaluate_regime_benchmark(samples=samples[:6], model_path=model_path)
    assert "summary" in report
    assert "overall_accuracy" in report["summary"]
    assert "per_class_metrics" in report["summary"]
    assert len(report["rows"]) == 6

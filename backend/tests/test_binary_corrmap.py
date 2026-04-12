from pathlib import Path

import torch

from scripts.build_regime_benchmark_refined_v2 import build_regime_benchmark_refined_v2
from scripts.eval_binary_corrmap_on_regime_benchmark import evaluate_binary_regime_benchmark


def test_binary_label_projection_from_refined_v2() -> None:
    samples = build_regime_benchmark_refined_v2()
    truth = [("quantum_frontier" if sample["benchmark_label"] == "quantum_frontier" else "classical_scalable") for sample in samples]
    assert len(samples) == 60
    assert truth.count("quantum_frontier") == 18
    assert truth.count("classical_scalable") == 42


def test_binary_eval_runs_if_model_exists() -> None:
    model_path = Path("backend/artifacts/binary_corrmap.pt")
    if not model_path.exists():
        return
    samples = build_regime_benchmark_refined_v2()
    report = evaluate_binary_regime_benchmark(samples=samples[:6], model_path=model_path)
    assert "summary" in report
    assert "overall_accuracy" in report["summary"]
    assert len(report["rows"]) == 6

from pathlib import Path

from scripts.build_regime_benchmark_refined_v2 import build_regime_benchmark_refined_v2
from scripts.eval_corrmap_on_refined_benchmark import evaluate_refined_benchmark


def test_refined_v2_regime_benchmark_has_balanced_labels() -> None:
    samples = build_regime_benchmark_refined_v2()
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample["benchmark_label"]] = counts.get(sample["benchmark_label"], 0) + 1
    assert len(samples) == 60
    assert counts == {
        "mean_field": 20,
        "scalable_classical": 22,
        "quantum_frontier": 18,
    }


def test_refined_v2_eval_runs_with_current_hybrid_model() -> None:
    model_path = Path("backend/artifacts/hybrid_corrmap_augmented.pt")
    if not model_path.exists():
        return
    samples = build_regime_benchmark_refined_v2()
    report = evaluate_refined_benchmark(samples=samples[:6], model_path=model_path)
    assert "summary" in report
    assert "hard_accuracy" in report["summary"]
    assert "soft_accuracy" in report["summary"]
    assert len(report["rows"]) == 6

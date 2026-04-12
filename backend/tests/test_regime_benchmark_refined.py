from pathlib import Path

from scripts.build_regime_benchmark_refined import build_regime_benchmark_refined
from scripts.eval_corrmap_on_refined_benchmark import evaluate_refined_benchmark


def test_refined_regime_benchmark_has_balanced_labels_and_soft_probs() -> None:
    samples = build_regime_benchmark_refined()
    counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for sample in samples:
        counts[sample["benchmark_label"]] = counts.get(sample["benchmark_label"], 0) + 1
        confidence_counts[sample["benchmark_confidence"]] = confidence_counts.get(sample["benchmark_confidence"], 0) + 1
        probs = sample["benchmark_label_probs"]
        assert set(probs) == {"mean_field", "scalable_classical", "quantum_frontier"}
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    assert len(samples) == 60
    assert counts == {
        "mean_field": 20,
        "scalable_classical": 20,
        "quantum_frontier": 20,
    }
    assert confidence_counts == {
        "high": 48,
        "medium": 12,
    }


def test_refined_regime_benchmark_eval_runs_with_current_hybrid_model() -> None:
    model_path = Path("backend/artifacts/hybrid_corrmap_augmented.pt")
    if not model_path.exists():
        return
    samples = build_regime_benchmark_refined()
    report = evaluate_refined_benchmark(samples=samples[:6], model_path=model_path)
    assert "summary" in report
    assert "hard_accuracy" in report["summary"]
    assert "soft_accuracy" in report["summary"]
    assert len(report["rows"]) == 6

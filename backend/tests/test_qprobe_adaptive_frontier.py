from __future__ import annotations

from scripts.benchmark_qprobe_adaptive_frontier import _difficulty_band, _planner_metrics


def test_difficulty_band_thresholds() -> None:
    assert _difficulty_band(1) == "easy"
    assert _difficulty_band(4) == "easy"
    assert _difficulty_band(5) == "edge"
    assert _difficulty_band(6) == "edge"
    assert _difficulty_band(7) == "frontier"


def test_metrics_summary_shapes() -> None:
    rows = [
        {"bounded_regret": 0, "bounded_oracle_safe": True, "bounded_runtime_stop": True, "subset_budget": 31},
        {"bounded_regret": 1, "bounded_oracle_safe": True, "bounded_runtime_stop": False, "subset_budget": 63},
    ]
    metrics = _planner_metrics(rows, regret_key="bounded_regret", safe_key="bounded_oracle_safe", stop_key="bounded_runtime_stop")
    assert metrics["count"] == 2
    assert metrics["within_optimal"] == 0.5
    assert metrics["within_plus_one"] == 1.0
    assert metrics["oracle_safe_rate"] == 1.0

from __future__ import annotations

from scripts.benchmark_qprobe_adaptive_stack import _exact_metrics, _frontier_metrics, _zone


def test_zone_thresholds() -> None:
    assert _zone(4, 6) == "overlap"
    assert _zone(5, 6) == "edge"
    assert _zone(6, 6) == "edge"
    assert _zone(7, 6) == "frontier"


def test_exact_metrics_shape() -> None:
    rows = [
        {
            "exact_ran": True,
            "bounded_regret": 0,
            "bounded_oracle_safe": True,
            "bounded_runtime_stop": True,
            "subset_budget": 31,
        },
        {
            "exact_ran": True,
            "bounded_regret": 1,
            "bounded_oracle_safe": False,
            "bounded_runtime_stop": False,
            "subset_budget": 63,
        },
    ]
    metrics = _exact_metrics(rows, "bounded_regret", "bounded_oracle_safe", "bounded_runtime_stop")
    assert metrics["count"] == 2
    assert metrics["within_optimal"] == 0.5
    assert metrics["within_plus_one"] == 1.0


def test_frontier_metrics_shape() -> None:
    rows = [
        {
            "exact_ran": False,
            "bounded_gap_to_lb": 1,
            "bounded_oracle_safe": True,
            "bounded_runtime_stop": True,
            "subset_budget": 127,
        },
        {
            "exact_ran": False,
            "bounded_gap_to_lb": 2,
            "bounded_oracle_safe": False,
            "bounded_runtime_stop": False,
            "subset_budget": 255,
        },
    ]
    metrics = _frontier_metrics(rows, "bounded_gap_to_lb", "bounded_oracle_safe", "bounded_runtime_stop")
    assert metrics["count"] == 2
    assert metrics["within_lb_plus_one"] == 0.5
    assert metrics["within_lb_plus_two"] == 1.0

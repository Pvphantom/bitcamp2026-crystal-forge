from __future__ import annotations

from scripts.eval_qprobe_general_model import _metrics_for_rows


def test_metrics_for_rows_tracks_undercompression_and_false_safe() -> None:
    rows = [
        {
            "predicted_cost": 1,
            "true_cost": 2,
            "predicted_success": True,
            "true_success": True,
            "predicted_error": 0.01,
            "true_error": 0.02,
        },
        {
            "predicted_cost": 4,
            "true_cost": 2,
            "predicted_success": True,
            "true_success": False,
            "predicted_error": 0.03,
            "true_error": 0.01,
        },
        {
            "predicted_cost": 2,
            "true_cost": 2,
            "predicted_success": False,
            "true_success": False,
            "predicted_error": 0.02,
            "true_error": 0.02,
        },
    ]

    metrics = _metrics_for_rows(rows)
    assert metrics["num_samples"] == 3
    assert metrics["cost_accuracy"] == 1 / 3
    assert metrics["success_accuracy"] == 2 / 3
    assert metrics["undercompression_rate"] == 1 / 3
    assert metrics["unsafe_undercompression_rate"] == 1 / 3
    assert metrics["overconservative_rate"] == 1 / 3
    assert metrics["false_safe_rate"] == 1 / 2

from pathlib import Path

import torch

from app.ml.infer import RoutingInferenceEngine
from scripts.eval_routing_model import evaluate_samples
from scripts.train_routing_model import train


def test_eval_routing_model_reports_grouped_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "routing_eval_dataset.pt"
    model_path = tmp_path / "routing_eval_model.pt"
    metrics_path = tmp_path / "routing_eval_metrics.json"

    samples = []
    for idx in range(8):
        samples.append(
            {
                "features": torch.full((22,), 0.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "mean_field",
                "reference_quality": "strong",
                "problem_metadata": {"family": "hubbard", "Lx": 2, "Ly": 2},
            }
        )
        samples.append(
            {
                "features": torch.full((22,), 3.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "scalable_classical",
                "reference_quality": "strong",
                "problem_metadata": {"family": "hubbard", "Lx": 2, "Ly": 3},
            }
        )
        samples.append(
            {
                "features": torch.full((22,), -3.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "quantum_frontier",
                "reference_quality": "strong",
                "problem_metadata": {"family": "tfim", "Lx": 2, "Ly": 2},
            }
        )
    torch.save(samples, dataset_path)

    train(
        dataset_path=dataset_path,
        model_out=model_path,
        metrics_out=metrics_path,
        epochs=60,
        batch_size=4,
        learning_rate=1e-3,
        allow_weak_labels=False,
        include_uncertain=False,
    )
    inference = RoutingInferenceEngine(model_path=model_path)
    metrics = evaluate_samples(samples, inference=inference)

    assert metrics["overall"]["coverage"] is not None
    assert metrics["overall"]["coverage"] > 0.0
    assert "strong" in metrics["by_reference_quality"]
    assert "hubbard" in metrics["by_family"]
    assert "tfim" in metrics["by_family"]
    assert "2x2" in metrics["by_lattice"]
    assert "2x3" in metrics["by_lattice"]
    assert "mean_field" in metrics["by_truth_label"]
    assert "scalable_classical" in metrics["by_truth_label"]

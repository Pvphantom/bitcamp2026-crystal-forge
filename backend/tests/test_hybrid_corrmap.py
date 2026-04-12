from pathlib import Path

import torch

from app.ml.infer import HybridCorrMapInferenceEngine
from app.ml.schema import INTRINSIC_RISK_LABELS, ROUTING_LABELS
from scripts.train_hybrid_corrmap import train


def test_hybrid_training_and_inference_pipeline(tmp_path: Path) -> None:
    routing_dataset = tmp_path / "routing.pt"
    intrinsic_dataset = tmp_path / "intrinsic.pt"
    model_path = tmp_path / "hybrid.pt"
    metrics_path = tmp_path / "hybrid.json"

    routing_samples = []
    for idx in range(8):
        routing_samples.append(
            {
                "features": torch.full((22,), 0.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "mean_field",
                "reference_quality": "strong",
                "problem_metadata": {"nsites": 4},
            }
        )
        routing_samples.append(
            {
                "features": torch.full((22,), 3.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "scalable_classical",
                "reference_quality": "strong",
                "problem_metadata": {"nsites": 4},
            }
        )
    intrinsic_samples = []
    for idx in range(8):
        intrinsic_samples.append(
            {
                "features": torch.full((22,), 0.0 + 0.05 * idx, dtype=torch.float32),
                "intrinsic_label": "stable_classical",
            }
        )
        intrinsic_samples.append(
            {
                "features": torch.full((22,), 6.0 + 0.05 * idx, dtype=torch.float32),
                "intrinsic_label": "frontier_or_uncertain",
            }
        )
        intrinsic_samples.append(
            {
                "features": torch.full((22,), 2.0 + 0.05 * idx, dtype=torch.float32),
                "intrinsic_label": "fragile_classical",
            }
        )
    torch.save(routing_samples, routing_dataset)
    torch.save(intrinsic_samples, intrinsic_dataset)

    metrics = train(
        routing_dataset_path=routing_dataset,
        intrinsic_dataset_path=intrinsic_dataset,
        model_out=model_path,
        metrics_out=metrics_path,
        epochs=60,
        batch_size=4,
        learning_rate=1e-3,
    )
    assert model_path.exists()
    assert metrics["routing_test_accuracy"] >= 0.0
    assert metrics["intrinsic_test_accuracy"] >= 0.0

    inference = HybridCorrMapInferenceEngine(model_path=model_path)
    stable = inference.predict(torch.full((22,), 0.1, dtype=torch.float32))
    frontier = inference.predict(torch.full((22,), 6.1, dtype=torch.float32))
    assert stable is not None and frontier is not None
    assert stable["label"] in ROUTING_LABELS
    assert stable["intrinsic_label"] in INTRINSIC_RISK_LABELS
    assert frontier["intrinsic_label"] in INTRINSIC_RISK_LABELS

from pathlib import Path

import torch

from app.ml.infer import RoutingInferenceEngine
from app.ml.routing_model import RoutingMLP
from app.ml.schema import ROUTING_LABELS
from scripts.train_routing_model import train


def test_routing_model_output_shapes() -> None:
    model = RoutingMLP()
    outputs = model(torch.zeros((4, 22), dtype=torch.float32))
    assert outputs["route_logits"].shape == (4, 4)
    assert outputs["confidence_logit"].shape == (4,)


def test_routing_training_and_inference_pipeline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "routing_dataset.pt"
    model_path = tmp_path / "routing_model.pt"
    metrics_path = tmp_path / "routing_metrics.json"

    samples = []
    route_centers = {
        "mean_field": 0.0,
        "scalable_classical": 3.0,
        "quantum_frontier": -3.0,
    }
    for label, center in route_centers.items():
        for idx in range(8):
            features = torch.full((22,), center, dtype=torch.float32)
            features = features + 0.05 * idx
            samples.append(
                {
                    "features": features,
                    "route_label": label,
                    "reference_quality": "strong",
                }
            )
    samples.append(
        {
            "features": torch.ones(22, dtype=torch.float32),
            "route_label": "uncertain",
            "reference_quality": "weak",
        }
    )
    torch.save(samples, dataset_path)

    metrics = train(
        dataset_path=dataset_path,
        model_out=model_path,
        metrics_out=metrics_path,
        epochs=60,
        batch_size=4,
        learning_rate=1e-3,
        allow_weak_labels=False,
        include_uncertain=False,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["num_raw_samples"] == len(samples)
    assert metrics["num_filtered_samples"] == len(samples) - 1
    assert metrics["allowed_reference_qualities"] == ["strong"]
    assert metrics["test"]["coverage"] > 0.0

    inference = RoutingInferenceEngine(model_path=model_path)
    prediction = inference.predict(torch.full((22,), 3.05, dtype=torch.float32))
    assert prediction is not None
    assert prediction["available"] is True
    assert prediction["label"] in ROUTING_LABELS
    assert prediction["abstained"] is False

    ood_prediction = inference.predict(torch.full((22,), 20.0, dtype=torch.float32))
    assert ood_prediction is not None
    assert ood_prediction["label"] == "uncertain"
    assert ood_prediction["abstained"] is True
    assert ood_prediction["abstain_reason"] == "ood_distance"

import torch

from app.ml.schema import DEFAULT_TRUST_DATASET
from app.ml.trust_model import TrustMLP


def test_trust_model_output_shapes() -> None:
    model = TrustMLP()
    outputs = model(torch.zeros((4, 22), dtype=torch.float32))
    assert outputs["risk_logits"].shape == (4, 3)
    assert outputs["error_pred"].shape == (4,)


def test_trust_dataset_artifact_exists_and_is_nonempty() -> None:
    samples = torch.load(DEFAULT_TRUST_DATASET, map_location="cpu")
    assert len(samples) > 0
    assert samples[0]["features"].shape == (22,)
    assert samples[0]["risk_label"] in {"safe", "warning", "unsafe"}

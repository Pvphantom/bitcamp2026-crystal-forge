import torch

from app.ml.qprobe_model import QProbeMLP
from app.ml.schema import DEFAULT_QPROBE_DATASET


def test_qprobe_model_output_shapes() -> None:
    model = QProbeMLP()
    outputs = model(torch.zeros((4, 15), dtype=torch.float32))
    assert outputs["cost_logits"].shape == (4, 4)
    assert outputs["success_logits"].shape == (4, 2)
    assert outputs["error_pred"].shape == (4,)


def test_qprobe_dataset_artifact_exists_and_is_nonempty() -> None:
    samples = torch.load(DEFAULT_QPROBE_DATASET, map_location="cpu")
    assert len(samples) > 0
    assert samples[0]["features"].shape == (15,)

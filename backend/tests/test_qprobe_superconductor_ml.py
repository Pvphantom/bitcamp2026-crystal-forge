from __future__ import annotations

from app.ml.qprobe_superconductor_features import qprobe_superconductor_feature_dim
from scripts.data_gen_qprobe_superconductor_ml import build_dataset


def test_superconductor_ml_dataset_builds() -> None:
    samples = build_dataset(quick=True, node_budget=32)
    assert samples
    assert samples[0]["features"].shape[0] == qprobe_superconductor_feature_dim()
    assert "bundle_name" in samples[0]["metadata"]

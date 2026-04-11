from pathlib import Path

import torch

from app.ml.infer import MetricsReader, PhaseInferenceEngine
from app.ml.schema import PHASE_TO_INDEX, build_graph_sample, classify_phase_rule


def test_classification_rules_cover_expected_regions() -> None:
    assert classify_phase_rule(1.0, 0.9, 0.05) == "Metal"
    assert classify_phase_rule(8.0, 1.0, 0.05) == "Mott Insulator"
    assert classify_phase_rule(5.0, 1.0, 0.45) == "Antiferromagnet"
    assert classify_phase_rule(3.0, 0.8, 0.1) == "Singlet-rich"


def test_graph_sample_builder_shapes() -> None:
    sample = build_graph_sample(
        Lx=2,
        Ly=2,
        site_features=[[1, 0, 0, 0.5, 1], [0, 1, 0, -0.5, -1], [0, 1, 0, -0.5, -1], [1, 0, 0, 0.5, 1]],
        bond_strengths={(0, 1): -0.25, (0, 2): -0.25, (1, 3): -0.25, (2, 3): -0.25},
        global_feats=[4.0, 2.0, 4.0],
        label="Antiferromagnet",
        metadata={"sample_id": "demo"},
        max_nodes=4,
    ).to_dict()
    assert sample["nodes"].shape == (4, 5)
    assert sample["adjacency"].shape == (4, 4)
    assert sample["edge_attr"].shape == (4, 4, 1)
    assert sample["label"] == PHASE_TO_INDEX["Antiferromagnet"]


def test_inference_engine_returns_none_without_model() -> None:
    engine = PhaseInferenceEngine(model_path=Path("backend/artifacts/definitely_missing.pt"))
    result = engine.predict(
        {
            "nodes": torch.zeros((4, 5)),
            "adjacency": torch.zeros((4, 4)),
            "edge_attr": torch.zeros((4, 4, 1)),
            "node_mask": torch.ones(4),
            "global_feats": torch.zeros(3),
        }
    )
    assert result is None


def test_metrics_reader_reports_missing_artifacts_cleanly(tmp_path: Path) -> None:
    metrics = MetricsReader(
        metrics_path=tmp_path / "missing_metrics.json",
        model_path=tmp_path / "missing_model.pt",
    ).summary()
    assert metrics["available"] is False
    assert metrics["model_loaded"] is False
    assert metrics["phase_labels"] == ["Metal", "Mott Insulator", "Antiferromagnet", "Singlet-rich"]
    assert metrics["confusion_matrix"] == []

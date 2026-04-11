"""Inference helpers for the phase classifier."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import torch

from app.ml.model import HubbardPhaseGraphNet
from app.ml.schema import DEFAULT_MODEL_PATH, PHASE_LABELS


class PhaseInferenceEngine:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model: HubbardPhaseGraphNet | None = None
        self._loaded = False

    def is_available(self) -> bool:
        return self.model_path.exists()

    def status(self, *, source: str | None = None) -> dict[str, Any]:
        return {
            "source": source or ("trained-model" if self.is_available() else "fallback-rules"),
            "model_loaded": self.is_available(),
            "model_path": str(self.model_path),
        }

    def predict(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        model = self._load()
        model.eval()
        with torch.no_grad():
            logits = model(
                nodes=sample["nodes"].unsqueeze(0),
                adjacency=sample["adjacency"].unsqueeze(0),
                edge_attr=sample["edge_attr"].unsqueeze(0),
                node_mask=sample["node_mask"].unsqueeze(0),
                global_feats=sample["global_feats"].unsqueeze(0),
            )
            probs = torch.softmax(logits, dim=-1)[0]
        label_index = int(torch.argmax(probs).item())
        probabilities = {
            PHASE_LABELS[index]: float(probs[index].item())
            for index in range(len(PHASE_LABELS))
        }
        return {
            "label": PHASE_LABELS[label_index],
            "confidence": float(probs[label_index].item()),
            "probabilities": probabilities,
            "model_status": self.status(source="trained-model"),
        }

    def _load(self) -> HubbardPhaseGraphNet:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = HubbardPhaseGraphNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._model = model
        self._loaded = True
        return model


class MetricsReader:
    def __init__(
        self,
        metrics_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.metrics_path = metrics_path or (DEFAULT_MODEL_PATH.parent / "gnn_metrics.json")
        self.model_path = model_path or DEFAULT_MODEL_PATH

    def summary(self) -> dict[str, Any]:
        base = {
            "available": False,
            "model_loaded": self.model_path.exists(),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
            "phase_labels": PHASE_LABELS,
            "train_accuracy": None,
            "val_accuracy": None,
            "test_accuracy": None,
            "cross_lattice_accuracy": None,
            "confusion_matrix": [],
            "cross_lattice_confusion_matrix": [],
        }
        if not self.metrics_path.exists():
            return base

        payload = json.loads(self.metrics_path.read_text())
        return {
            **base,
            "available": True,
            "train_accuracy": payload.get("train_accuracy"),
            "val_accuracy": payload.get("val_accuracy"),
            "test_accuracy": payload.get("test_accuracy"),
            "cross_lattice_accuracy": payload.get("cross_lattice_accuracy"),
            "confusion_matrix": payload.get("confusion_matrix", []),
            "cross_lattice_confusion_matrix": payload.get("cross_lattice_confusion_matrix", []),
        }

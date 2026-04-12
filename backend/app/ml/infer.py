"""Inference helpers for the phase classifier."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import torch

from app.ml.model import HubbardPhaseGraphNet
from app.ml.hybrid_corrmap_model import HybridCorrMapMLP
from app.ml.qprobe_model import QProbeMLP
from app.ml.routing_model import RoutingMLP
from app.ml.schema import (
    DEFAULT_HYBRID_CORRMAP_METRICS_PATH,
    DEFAULT_HYBRID_CORRMAP_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QPROBE_METRICS_PATH,
    DEFAULT_QPROBE_MODEL_PATH,
    DEFAULT_ROUTING_METRICS_PATH,
    DEFAULT_ROUTING_MODEL_PATH,
    DEFAULT_TRUST_METRICS_PATH,
    DEFAULT_TRUST_MODEL_PATH,
    PHASE_LABELS,
    ROUTING_LABELS,
    TRUST_LABELS,
)
from app.ml.trust_model import TrustMLP


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


class QProbeInferenceEngine:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_QPROBE_MODEL_PATH
        self._model: QProbeMLP | None = None
        self._loaded = False
        self._cost_classes: list[int] = [1, 2, 4, 6]
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def predict(self, features: torch.Tensor | Any) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        model = self._load()
        assert self._feature_mean is not None and self._feature_std is not None
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        normalized = (feature_tensor - self._feature_mean) / self._feature_std
        model.eval()
        with torch.no_grad():
            outputs = model(normalized.unsqueeze(0))
            cost_index = int(torch.argmax(outputs["cost_logits"], dim=-1).item())
            success_index = int(torch.argmax(outputs["success_logits"], dim=-1).item())
            error_value = float(outputs["error_pred"].item())
        return {
            "available": True,
            "model_path": str(self.model_path),
            "predicted_cost": self._cost_classes[cost_index],
            "predicted_success": bool(success_index),
            "predicted_error": error_value,
        }

    def _load(self) -> QProbeMLP:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = QProbeMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._cost_classes = list(checkpoint["cost_classes"])
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"].clamp_min(1e-6)
        self._model = model
        self._loaded = True
        return model


class QProbeMetricsReader:
    def __init__(
        self,
        metrics_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.metrics_path = metrics_path or DEFAULT_QPROBE_METRICS_PATH
        self.model_path = model_path or DEFAULT_QPROBE_MODEL_PATH

    def summary(self) -> dict[str, Any]:
        base = {
            "available": False,
            "model_loaded": self.model_path.exists(),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
        }
        if not self.metrics_path.exists():
            return base
        payload = json.loads(self.metrics_path.read_text())
        return {
            **base,
            "available": True,
            "train_cost_accuracy": payload.get("train", {}).get("cost_accuracy"),
            "val_cost_accuracy": payload.get("val", {}).get("cost_accuracy"),
            "test_cost_accuracy": payload.get("test", {}).get("cost_accuracy"),
            "test_success_accuracy": payload.get("test", {}).get("success_accuracy"),
            "test_error_mae": payload.get("test", {}).get("error_mae"),
            "test_false_safe_rate": payload.get("test", {}).get("false_safe_rate"),
        }


class TrustInferenceEngine:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or DEFAULT_TRUST_MODEL_PATH
        self._model: TrustMLP | None = None
        self._loaded = False
        self._labels = TRUST_LABELS
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None
        self._safe_error_guard: float | None = None
        self._safe_prob_guard: float | None = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def predict(self, features: torch.Tensor | Any) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        try:
            model = self._load()
            assert self._feature_mean is not None and self._feature_std is not None
            feature_tensor = torch.as_tensor(features, dtype=torch.float32)
            if feature_tensor.shape[-1] != self._feature_mean.shape[-1]:
                return None
            normalized = (feature_tensor - self._feature_mean) / self._feature_std
            model.eval()
            with torch.no_grad():
                outputs = model(normalized.unsqueeze(0))
                probs = torch.softmax(outputs["risk_logits"], dim=-1)[0]
                label_index = int(torch.argmax(probs).item())
                predicted_error = float(outputs["error_pred"].item())
            if (
                self._safe_error_guard is not None
                and self._safe_prob_guard is not None
                and label_index == self._labels.index("safe")
                and (predicted_error > self._safe_error_guard or float(probs[label_index].item()) < self._safe_prob_guard)
            ):
                label_index = self._labels.index("warning")
            label = self._labels[label_index]
            return {
                "available": True,
                "model_path": str(self.model_path),
                "label": label,
                "confidence": float(probs[label_index].item()),
                "predicted_max_abs_error": predicted_error,
                "recommended_action": _trust_action(label),
            }
        except Exception:
            return None

    def _load(self) -> TrustMLP:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = TrustMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._labels = list(checkpoint["labels"])
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"].clamp_min(1e-6)
        self._safe_error_guard = checkpoint.get("safe_error_guard")
        self._safe_prob_guard = checkpoint.get("safe_prob_guard")
        self._model = model
        self._loaded = True
        return model


class TrustMetricsReader:
    def __init__(
        self,
        metrics_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.metrics_path = metrics_path or DEFAULT_TRUST_METRICS_PATH
        self.model_path = model_path or DEFAULT_TRUST_MODEL_PATH

    def summary(self) -> dict[str, Any]:
        base = {
            "available": False,
            "model_loaded": self.model_path.exists(),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
            "labels": TRUST_LABELS,
            "train_risk_accuracy": None,
            "val_risk_accuracy": None,
            "test_risk_accuracy": None,
            "test_error_mae": None,
            "test_false_safe_rate": None,
            "confusion_matrix": [],
        }
        if not self.metrics_path.exists():
            return base
        payload = json.loads(self.metrics_path.read_text())
        return {
            **base,
            "available": True,
            "labels": payload.get("labels", TRUST_LABELS),
            "train_risk_accuracy": payload.get("train", {}).get("risk_accuracy"),
            "val_risk_accuracy": payload.get("val", {}).get("risk_accuracy"),
            "test_risk_accuracy": payload.get("test", {}).get("risk_accuracy"),
            "test_error_mae": payload.get("test", {}).get("error_mae"),
            "test_false_safe_rate": payload.get("test", {}).get("false_safe_rate"),
            "confusion_matrix": payload.get("test", {}).get("confusion_matrix", []),
            "cross_lattice_risk_accuracy": payload.get("cross_lattice", {}).get("risk_accuracy"),
            "cross_lattice_false_safe_rate": payload.get("cross_lattice", {}).get("false_safe_rate"),
        }


class RoutingInferenceEngine:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_ROUTING_MODEL_PATH
        self._model: RoutingMLP | None = None
        self._loaded = False
        self._labels = ROUTING_LABELS
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None
        self._confidence_threshold: float | None = None
        self._ood_distance_threshold: float | None = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def predict(self, features: torch.Tensor | Any) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        try:
            model = self._load()
            assert self._feature_mean is not None and self._feature_std is not None
            feature_tensor = torch.as_tensor(features, dtype=torch.float32)
            if feature_tensor.shape[-1] != self._feature_mean.shape[-1]:
                return None
            normalized = (feature_tensor - self._feature_mean) / self._feature_std
            model.eval()
            with torch.no_grad():
                outputs = model(normalized.unsqueeze(0))
                probs = torch.softmax(outputs["route_logits"], dim=-1)[0]
                label_index = int(torch.argmax(probs).item())
                confidence = float(probs[label_index].item())
                distance = float(torch.linalg.norm(normalized).item())
            label = self._labels[label_index]
            abstained = False
            abstain_reason = None
            if self._ood_distance_threshold is not None and distance > self._ood_distance_threshold:
                abstained = True
                abstain_reason = "ood_distance"
            elif self._confidence_threshold is not None and confidence < self._confidence_threshold:
                abstained = True
                abstain_reason = "low_confidence"
            if abstained:
                label = "uncertain"
            return {
                "available": True,
                "model_path": str(self.model_path),
                "label": label,
                "confidence": confidence,
                "recommended_action": _routing_action(label),
                "candidate_scores": {
                    self._labels[index]: float(probs[index].item())
                    for index in range(len(self._labels))
                },
                "abstained": abstained,
                "abstain_reason": abstain_reason,
                "ood_distance": distance,
            }
        except Exception:
            return None

    def _load(self) -> RoutingMLP:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = RoutingMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._labels = list(checkpoint["labels"])
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"].clamp_min(1e-6)
        self._confidence_threshold = checkpoint.get("confidence_threshold")
        self._ood_distance_threshold = checkpoint.get("ood_distance_threshold")
        self._model = model
        self._loaded = True
        return model


class RoutingMetricsReader:
    def __init__(
        self,
        metrics_path: Path | str | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_ROUTING_METRICS_PATH
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_ROUTING_MODEL_PATH

    def summary(self) -> dict[str, Any]:
        base = {
            "available": False,
            "model_loaded": self.model_path.exists(),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
            "labels": ROUTING_LABELS,
            "train_route_accuracy": None,
            "val_route_accuracy": None,
            "test_route_accuracy": None,
            "test_coverage": None,
            "test_abstention_rate": None,
            "test_false_mean_field_rate": None,
            "confidence_threshold": None,
            "ood_distance_threshold": None,
            "confusion_matrix": [],
        }
        if not self.metrics_path.exists():
            return base
        payload = json.loads(self.metrics_path.read_text())
        return {
            **base,
            "available": True,
            "labels": payload.get("labels", ROUTING_LABELS),
            "train_route_accuracy": payload.get("train", {}).get("route_accuracy"),
            "val_route_accuracy": payload.get("val", {}).get("route_accuracy"),
            "test_route_accuracy": payload.get("test", {}).get("route_accuracy"),
            "test_coverage": payload.get("test", {}).get("coverage"),
            "test_abstention_rate": payload.get("test", {}).get("abstention_rate"),
            "test_false_mean_field_rate": payload.get("test", {}).get("false_mean_field_rate"),
            "confidence_threshold": payload.get("confidence_threshold"),
            "ood_distance_threshold": payload.get("ood_distance_threshold"),
            "confusion_matrix": payload.get("test", {}).get("confusion_matrix", []),
        }


class HybridCorrMapInferenceEngine:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_HYBRID_CORRMAP_MODEL_PATH
        self._model: HybridCorrMapMLP | None = None
        self._loaded = False
        self._route_labels = ROUTING_LABELS
        self._intrinsic_labels: list[str] = []
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None
        self._confidence_threshold: float | None = None
        self._ood_distance_threshold: float | None = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def predict(self, features: torch.Tensor | Any) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        try:
            model = self._load()
            assert self._feature_mean is not None and self._feature_std is not None
            feature_tensor = torch.as_tensor(features, dtype=torch.float32)
            if feature_tensor.shape[-1] != self._feature_mean.shape[-1]:
                return None
            normalized = (feature_tensor - self._feature_mean) / self._feature_std
            model.eval()
            with torch.no_grad():
                outputs = model(normalized.unsqueeze(0))
                route_probs = torch.softmax(outputs["route_logits"], dim=-1)[0]
                intrinsic_probs = torch.softmax(outputs["intrinsic_logits"], dim=-1)[0]
                route_index = int(torch.argmax(route_probs).item())
                intrinsic_index = int(torch.argmax(intrinsic_probs).item())
                confidence = float(route_probs[route_index].item())
                intrinsic_confidence = float(intrinsic_probs[intrinsic_index].item())
                distance = float(torch.linalg.norm(normalized).item())
            route_label = self._route_labels[route_index]
            intrinsic_label = self._intrinsic_labels[intrinsic_index]
            abstained = False
            abstain_reason = None
            if self._ood_distance_threshold is not None and distance > self._ood_distance_threshold:
                abstained = True
                abstain_reason = "ood_distance"
            elif self._confidence_threshold is not None and confidence < self._confidence_threshold:
                abstained = True
                abstain_reason = "low_confidence"
            elif intrinsic_label == "frontier_or_uncertain":
                abstained = True
                abstain_reason = "intrinsic_risk_guard"
            elif route_label == "mean_field" and intrinsic_label != "stable_classical":
                abstained = True
                abstain_reason = "intrinsic_risk_guard"
            final_route = "uncertain" if abstained else route_label
            return {
                "available": True,
                "model_path": str(self.model_path),
                "label": final_route,
                "confidence": confidence,
                "recommended_action": _routing_action(final_route),
                "candidate_scores": {
                    self._route_labels[index]: float(route_probs[index].item())
                    for index in range(len(self._route_labels))
                },
                "abstained": abstained,
                "abstain_reason": abstain_reason,
                "ood_distance": distance,
                "intrinsic_label": intrinsic_label,
                "intrinsic_score": intrinsic_confidence,
                "intrinsic_scores": {
                    self._intrinsic_labels[index]: float(intrinsic_probs[index].item())
                    for index in range(len(self._intrinsic_labels))
                },
            }
        except Exception:
            return None

    def _load(self) -> HybridCorrMapMLP:
        if self._loaded and self._model is not None:
            return self._model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model = HybridCorrMapMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self._route_labels = list(checkpoint["route_labels"])
        self._intrinsic_labels = list(checkpoint["intrinsic_labels"])
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"].clamp_min(1e-6)
        self._confidence_threshold = checkpoint.get("confidence_threshold")
        self._ood_distance_threshold = checkpoint.get("ood_distance_threshold")
        self._model = model
        self._loaded = True
        return model


class HybridCorrMapMetricsReader:
    def __init__(
        self,
        metrics_path: Path | str | None = None,
        model_path: Path | str | None = None,
    ) -> None:
        self.metrics_path = Path(metrics_path) if metrics_path is not None else DEFAULT_HYBRID_CORRMAP_METRICS_PATH
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_HYBRID_CORRMAP_MODEL_PATH

    def summary(self) -> dict[str, Any]:
        base = {
            "available": False,
            "model_loaded": self.model_path.exists(),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
        }
        if not self.metrics_path.exists():
            return base
        payload = json.loads(self.metrics_path.read_text())
        return {
            **base,
            "available": True,
            **payload,
        }


def _trust_action(label: str) -> str:
    if label == "safe":
        return "cheap_solver_ok"
    if label == "warning":
        return "check_exact_or_stronger_solver"
    return "escalate_to_exact_or_advanced_method"


def _routing_action(label: str) -> str:
    if label == "mean_field":
        return "use_mean_field"
    if label == "scalable_classical":
        return "use_scalable_classical_solver"
    if label == "quantum_frontier":
        return "use_quantum_solver"
    return "abstain_or_collect_stronger_evidence"

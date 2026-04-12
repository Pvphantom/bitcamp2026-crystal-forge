from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.analysis.general_tractability_features import GeneralTractabilityFeaturesReport
from app.analysis.runtime_intrinsic_corrmap import RuntimeIntrinsicCorrMapReport


@dataclass(frozen=True)
class PhysicsFirstCorrMapConfig:
    mean_field_weight: float = 1.0
    scalable_weight: float = 1.0
    quantum_weight: float = 1.0
    mean_field_threshold: float = 0.62
    quantum_threshold: float = 0.62
    mean_field_margin: float = 0.05
    quantum_margin: float = 0.05


@dataclass(frozen=True)
class PhysicsFirstCorrMapReport:
    label: str
    candidate_scores: dict[str, float]
    reasons: list[str]
    mean_field_score: float
    scalable_classical_score: float
    quantum_frontier_score: float


def score_physics_first_corrmap(
    *,
    runtime: RuntimeIntrinsicCorrMapReport,
    general: GeneralTractabilityFeaturesReport,
    config: PhysicsFirstCorrMapConfig | None = None,
) -> PhysicsFirstCorrMapReport:
    cfg = config or PhysicsFirstCorrMapConfig()
    physical = runtime.physical_tractability
    intrinsic = runtime.assessment

    mean_field_score = _clip01(
        cfg.mean_field_weight
        * (
            0.30 * physical.mean_field_plausibility
            + 0.20 * general.factorization_reserve
            + 0.15 * general.spatial_uniformity
            + 0.15 * general.fixed_point_stiffness
            + 0.10 * general.scale_transfer_stability
            + 0.10 * general.response_linearity
        )
    )
    quantum_frontier_score = _clip01(
        cfg.quantum_weight
        * (
            0.30 * physical.quantum_frontier_pressure
            + 0.20 * general.metastability_pressure
            + 0.15 * general.continuation_curvature
            + 0.10 * (1.0 - general.scale_transfer_stability)
            + 0.10 * general.classical_obstruction
            + 0.10 * general.correlation_load
            + 0.05 * (1.0 if intrinsic.label == "frontier_or_uncertain" else 0.0)
        )
    )
    scalable_classical_score = _clip01(
        cfg.scalable_weight
        * (
            0.30 * physical.scalable_classical_plausibility
            + 0.20 * (1.0 - mean_field_score)
            + 0.15 * (1.0 - quantum_frontier_score)
            + 0.15 * general.response_linearity
            + 0.10 * general.scale_transfer_stability
            + 0.10 * (1.0 - general.classical_obstruction)
        )
    )

    reasons: list[str] = []
    if physical.mean_field_plausibility >= 0.72:
        reasons.append("high_mean_field_plausibility")
    if physical.quantum_frontier_pressure >= 0.72:
        reasons.append("high_quantum_frontier_pressure")
    if general.factorization_reserve >= 0.65:
        reasons.append("high_factorization_reserve")
    if general.classical_obstruction >= 0.45:
        reasons.append("high_classical_obstruction")
    if general.metastability_pressure >= 0.45:
        reasons.append("high_metastability_pressure")

    label = _decide_label(
        mean_field_score=mean_field_score,
        scalable_classical_score=scalable_classical_score,
        quantum_frontier_score=quantum_frontier_score,
        config=cfg,
    )
    return PhysicsFirstCorrMapReport(
        label=label,
        candidate_scores={
            "mean_field": mean_field_score,
            "scalable_classical": scalable_classical_score,
            "quantum_frontier": quantum_frontier_score,
        },
        reasons=reasons + list(physical.reasons) + list(intrinsic.reasons),
        mean_field_score=mean_field_score,
        scalable_classical_score=scalable_classical_score,
        quantum_frontier_score=quantum_frontier_score,
    )


def apply_physics_first_overlay(
    *,
    base_prediction: dict[str, Any] | None,
    physics_report: PhysicsFirstCorrMapReport,
    runtime: RuntimeIntrinsicCorrMapReport,
) -> dict[str, Any]:
    label = physics_report.label
    if base_prediction is not None:
        candidate_scores = dict(base_prediction.get("candidate_scores", {}))
        candidate_scores.update(physics_report.candidate_scores)
    else:
        candidate_scores = dict(physics_report.candidate_scores)
    return {
        "available": True,
        "model_path": "physics_first_corrmap",
        "label": label,
        "confidence": max(candidate_scores.values()) if candidate_scores else 0.0,
        "recommended_action": _action(label),
        "candidate_scores": candidate_scores,
        "abstained": False,
        "abstain_reason": None,
        "intrinsic_label": runtime.assessment.label,
        "intrinsic_score": runtime.assessment.score,
        "intrinsic_reasons": list(runtime.assessment.reasons),
        "physical_first_reasons": list(physics_report.reasons),
        "mean_field_safety_score": runtime.mean_field_safety_score,
        "physical_tractability": {
            "mean_field_plausibility": runtime.physical_tractability.mean_field_plausibility,
            "scalable_classical_plausibility": runtime.physical_tractability.scalable_classical_plausibility,
            "quantum_frontier_pressure": runtime.physical_tractability.quantum_frontier_pressure,
        },
    }


def _decide_label(
    *,
    mean_field_score: float,
    scalable_classical_score: float,
    quantum_frontier_score: float,
    config: PhysicsFirstCorrMapConfig,
) -> str:
    if quantum_frontier_score >= config.quantum_threshold and quantum_frontier_score >= scalable_classical_score + config.quantum_margin:
        return "quantum_frontier"
    if mean_field_score >= config.mean_field_threshold and mean_field_score >= scalable_classical_score + config.mean_field_margin:
        return "mean_field"
    return "scalable_classical"


def _action(label: str) -> str:
    if label == "mean_field":
        return "use_mean_field"
    if label == "scalable_classical":
        return "escalate_to_stronger_classical_solver"
    if label == "quantum_frontier":
        return "route_to_quantum_solver"
    return "abstain_and_request_stronger_method"


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

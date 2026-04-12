from __future__ import annotations

from dataclasses import dataclass

from app.analysis.mf_hysteresis import MeanFieldHysteresisReport
from app.analysis.mf_ansatz_disagreement import MeanFieldAnsatzDisagreementReport
from app.analysis.mf_sensitivity import MeanFieldSensitivityReport
from app.analysis.mf_size_consistency import MeanFieldSizeConsistencyReport
from app.analysis.mf_stability import MeanFieldStabilityReport


@dataclass(frozen=True)
class IntrinsicRiskAssessment:
    label: str
    score: float
    reasons: list[str]


def assess_intrinsic_risk(
    *,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport | None = None,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport | None = None,
    hysteresis: MeanFieldHysteresisReport | None = None,
) -> IntrinsicRiskAssessment:
    score = 0.0
    reasons: list[str] = []

    if stability.converged_fraction < 1.0:
        score += 2.0
        reasons.append("non_converged_seeds")
    if stability.residual_max > 1e-5:
        score += 2.0
        reasons.append("high_residual")
    if stability.energy_span > 0.1:
        score += 1.5
        reasons.append("multi_seed_energy_instability")
    if stability.distinct_solution_count > 2:
        score += 1.5
        reasons.append("multiple_fixed_points")
    if max(stability.observable_spans.values(), default=0.0) > 0.1:
        score += 1.0
        reasons.append("observable_seed_instability")
    if sensitivity.observable_shift_max > 0.15:
        score += 1.5
        reasons.append("parameter_sensitivity")
    if sensitivity.energy_density_shift_max > 0.15:
        score += 1.0
        reasons.append("energy_density_sensitivity")
    if size_consistency is not None:
        if size_consistency.observable_shift_max > 0.12:
            score += 1.5
            reasons.append("size_inconsistency")
        if size_consistency.energy_density_shift > 0.08:
            score += 1.0
            reasons.append("energy_density_size_drift")
    if ansatz_disagreement is not None:
        if ansatz_disagreement.max_abs_gap > 0.12:
            score += 1.75
            reasons.append("ansatz_disagreement")
        elif ansatz_disagreement.risk_label == "warning":
            score += 0.75
            reasons.append("ansatz_disagreement")
        if ansatz_disagreement.energy_density_gap > 0.08:
            score += 1.0
            reasons.append("ansatz_energy_gap")
    if hysteresis is not None:
        if hysteresis.observable_gap_max > 0.12:
            score += 1.75
            reasons.append("hysteresis")
        if hysteresis.energy_density_gap > 0.08:
            score += 1.0
            reasons.append("hysteresis_energy_gap")

    if score >= 4.0:
        label = "frontier_or_uncertain"
    elif score >= 1.5:
        label = "fragile_classical"
    else:
        label = "stable_classical"
    return IntrinsicRiskAssessment(label=label, score=float(score), reasons=reasons)

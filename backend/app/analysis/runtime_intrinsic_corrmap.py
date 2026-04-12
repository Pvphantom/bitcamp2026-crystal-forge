from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.analysis.mf_hysteresis import MeanFieldHysteresisReport, analyze_mean_field_hysteresis
from app.analysis.intrinsic_risk import IntrinsicRiskAssessment, assess_intrinsic_risk
from app.analysis.physical_tractability import PhysicalTractabilityReport, assess_physical_tractability
from app.analysis.mf_ansatz_disagreement import (
    MeanFieldAnsatzDisagreementReport,
    analyze_mean_field_ansatz_disagreement,
)
from app.analysis.mf_sensitivity import MeanFieldSensitivityReport, analyze_mean_field_sensitivity
from app.analysis.mf_size_consistency import MeanFieldSizeConsistencyReport, analyze_mean_field_size_consistency
from app.analysis.mf_stability import MeanFieldStabilityReport, analyze_mean_field_stability
from app.domain.problem_spec import ProblemSpec
from app.solvers.base import SolverResult
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


@dataclass(frozen=True)
class RuntimeIntrinsicCorrMapReport:
    assessment: IntrinsicRiskAssessment
    stability: MeanFieldStabilityReport
    sensitivity: MeanFieldSensitivityReport
    size_consistency: MeanFieldSizeConsistencyReport
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport
    hysteresis: MeanFieldHysteresisReport
    physical_tractability: PhysicalTractabilityReport
    mean_field_safety_score: float


def analyze_runtime_intrinsic_corrmap(
    problem: ProblemSpec,
    *,
    cheap_result: SolverResult | None = None,
    num_seeds: int = 3,
    init_noise_scale: float = 0.02,
    perturbation_scale: float = 0.05,
) -> RuntimeIntrinsicCorrMapReport:
    if cheap_result is None:
        cheap_solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
        cheap_result = cheap_solver.solve(problem)
    stability = analyze_mean_field_stability(
        problem,
        num_seeds=num_seeds,
        init_noise_scale=init_noise_scale,
    )
    sensitivity = analyze_mean_field_sensitivity(problem, perturbation_scale=perturbation_scale)
    size_consistency = analyze_mean_field_size_consistency(problem)
    ansatz_disagreement = analyze_mean_field_ansatz_disagreement(problem)
    hysteresis = analyze_mean_field_hysteresis(problem, perturbation_scale=perturbation_scale)
    assessment = assess_intrinsic_risk(
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
    )
    physical_tractability = assess_physical_tractability(
        problem=problem,
        cheap_result=cheap_result,
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
    )
    assessment = _apply_regime_prior(problem, assessment, physical_tractability)
    mean_field_safety_score = _mean_field_safety_score(
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
        physical_tractability=physical_tractability,
    )
    return RuntimeIntrinsicCorrMapReport(
        assessment=assessment,
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
        physical_tractability=physical_tractability,
        mean_field_safety_score=mean_field_safety_score,
    )


def apply_runtime_intrinsic_overlay(
    prediction: dict[str, Any] | None,
    report: RuntimeIntrinsicCorrMapReport,
) -> dict[str, Any]:
    intrinsic_label = report.assessment.label
    intrinsic_score = report.assessment.score
    reasons = list(report.assessment.reasons)
    mean_field_safety_score = float(getattr(report, "mean_field_safety_score", 1.0))
    physical = getattr(report, "physical_tractability", None)
    if physical is None:
        physical = PhysicalTractabilityReport(
            mean_field_plausibility=1.0 if intrinsic_label == "stable_classical" else 0.4,
            scalable_classical_plausibility=0.7 if intrinsic_label == "fragile_classical" else 0.4,
            quantum_frontier_pressure=1.0 if intrinsic_label == "frontier_or_uncertain" else 0.0,
            sign_problem_risk=0.0,
            stoquasticity=0.0,
            factorization_proxy=1.0 if intrinsic_label == "stable_classical" else 0.4,
            interaction_pressure=0.0,
            critical_pressure=0.0,
            route_prior=(
                "mean_field"
                if intrinsic_label == "stable_classical"
                else "scalable_classical"
                if intrinsic_label == "fragile_classical"
                else "quantum_frontier"
            ),
            reasons=[],
        )

    if prediction is None:
        fallback_route = physical.route_prior
        return {
            "label": fallback_route,
            "recommended_action": _intrinsic_action(fallback_route),
            "candidate_scores": {},
            "abstained": fallback_route == "uncertain",
            "abstain_reason": "intrinsic_runtime_only" if fallback_route == "uncertain" else None,
            "intrinsic_label": intrinsic_label,
            "intrinsic_score": intrinsic_score,
            "intrinsic_reasons": reasons,
            "mean_field_safety_score": mean_field_safety_score,
            "physical_tractability": _physical_payload(physical),
        }

    route_label = str(prediction.get("label", "uncertain"))
    candidate_scores = dict(prediction.get("candidate_scores", {}))
    abstained = bool(prediction.get("abstained", False))
    abstain_reason = prediction.get("abstain_reason")

    if intrinsic_label == "frontier_or_uncertain":
        route_label = "quantum_frontier"
        abstained = False
        abstain_reason = "intrinsic_runtime_quantum_escalation"
    elif (
        physical.route_prior == "quantum_frontier"
        and route_label != "quantum_frontier"
        and (intrinsic_label == "frontier_or_uncertain" or physical.quantum_frontier_pressure >= 0.85)
    ):
        route_label = "quantum_frontier"
        abstained = False
        abstain_reason = "physical_quantum_prior"
    elif intrinsic_label == "fragile_classical" and route_label in {"mean_field", "uncertain", "quantum_frontier"}:
        route_label = "scalable_classical"
        abstained = False
        abstain_reason = "intrinsic_runtime_escalation"
    elif intrinsic_label == "stable_classical" and (
        route_label == "uncertain"
        or (abstained and candidate_scores.get("mean_field", 0.0) >= candidate_scores.get("scalable_classical", 0.0))
    ):
        if mean_field_safety_score >= 0.9 or physical.mean_field_plausibility >= 0.78:
            route_label = "mean_field"
            abstained = False
            abstain_reason = "intrinsic_runtime_recovery"
        else:
            route_label = "scalable_classical"
            abstained = False
            abstain_reason = "mean_field_safety_guard"
    elif route_label == "mean_field" and mean_field_safety_score < 0.9:
        route_label = "scalable_classical"
        abstained = False
        abstain_reason = "mean_field_safety_guard"
    elif route_label == "mean_field" and intrinsic_label != "stable_classical" and physical.mean_field_plausibility < 0.58:
        route_label = "scalable_classical"
        abstained = False
        abstain_reason = "physical_mean_field_guard"

    return {
        **prediction,
        "label": route_label,
        "recommended_action": _intrinsic_action(route_label),
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "intrinsic_label": intrinsic_label,
        "intrinsic_score": intrinsic_score,
        "intrinsic_reasons": reasons,
        "mean_field_safety_score": mean_field_safety_score,
        "ansatz_disagreement_max_abs_gap": report.ansatz_disagreement.max_abs_gap,
        "ansatz_disagreement_energy_density_gap": report.ansatz_disagreement.energy_density_gap,
        "hysteresis_observable_gap_max": report.hysteresis.observable_gap_max,
        "hysteresis_energy_density_gap": report.hysteresis.energy_density_gap,
        "physical_tractability": _physical_payload(physical),
    }


def _intrinsic_action(route_label: str) -> str:
    if route_label == "mean_field":
        return "cheap_solver_ok"
    if route_label == "quantum_frontier":
        return "route_to_quantum_solver"
    if route_label == "scalable_classical":
        return "escalate_to_stronger_classical_solver"
    return "abstain_and_request_stronger_method"


def _apply_regime_prior(
    problem: ProblemSpec,
    assessment: IntrinsicRiskAssessment,
    physical_tractability: PhysicalTractabilityReport,
) -> IntrinsicRiskAssessment:
    score = float(assessment.score)
    reasons = list(assessment.reasons)

    if problem.model_family == "hubbard":
        near_half_filled = abs(problem.mu - problem.U / 2.0) <= 0.25
        if problem.U >= 6.0 and near_half_filled:
            score += 1.25
            reasons.append("strong_coupling_half_filling")
    elif problem.model_family == "tfim":
        critical_proximity = abs(problem.g) <= 0.15 and abs(problem.h / max(problem.J, 1e-8) - 1.0) <= 0.15
        if critical_proximity:
            score += 1.25
            reasons.append("critical_line_proximity")

    score += 1.5 * max(0.0, physical_tractability.quantum_frontier_pressure - 0.7)
    score += 0.5 * max(0.0, physical_tractability.sign_problem_risk - 0.6)
    if physical_tractability.route_prior == "quantum_frontier":
        reasons.append("physical_quantum_prior")
    elif physical_tractability.route_prior == "mean_field":
        score -= 0.5
        reasons.append("physical_mean_field_prior")

    if score >= 4.0:
        label = "frontier_or_uncertain"
    elif score >= 1.5:
        label = "fragile_classical"
    else:
        label = "stable_classical"
    return IntrinsicRiskAssessment(label=label, score=score, reasons=reasons)


def _mean_field_safety_score(
    *,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
    physical_tractability: PhysicalTractabilityReport,
) -> float:
    penalties = [
        2.0 * max(0.0, 1.0 - float(stability.converged_fraction)),
        min(1.0, float(stability.residual_max) / 1e-4),
        min(1.0, float(stability.energy_span) / 0.1),
        min(1.0, max(stability.observable_spans.values(), default=0.0) / 0.08),
        min(1.0, float(sensitivity.observable_shift_max) / 0.12),
        min(1.0, float(sensitivity.energy_density_shift_max) / 0.12),
        min(1.0, float(size_consistency.observable_shift_max) / 0.12),
        min(1.0, float(size_consistency.energy_density_shift) / 0.08),
        min(1.0, float(ansatz_disagreement.max_abs_gap) / 0.12),
        min(1.0, float(ansatz_disagreement.energy_density_gap) / 0.08),
        min(1.0, float(hysteresis.observable_gap_max) / 0.12),
        min(1.0, float(hysteresis.energy_density_gap) / 0.08),
        min(1.0, max(0.0, 0.72 - float(physical_tractability.mean_field_plausibility)) / 0.72),
        min(1.0, float(physical_tractability.quantum_frontier_pressure) / 0.9),
    ]
    weighted_penalty = (
        1.5 * penalties[0]
        + 1.2 * penalties[1]
        + 0.8 * penalties[2]
        + 0.7 * penalties[3]
        + 0.8 * penalties[4]
        + 0.5 * penalties[5]
        + 0.8 * penalties[6]
        + 0.5 * penalties[7]
        + 1.0 * penalties[8]
        + 0.6 * penalties[9]
        + 0.8 * penalties[10]
        + 0.4 * penalties[11]
        + 1.0 * penalties[12]
        + 0.8 * penalties[13]
    ) / 11.4
    return float(max(0.0, min(1.0, 1.0 - weighted_penalty)))


def _physical_payload(report: PhysicalTractabilityReport) -> dict[str, float | str | list[str]]:
    return {
        "mean_field_plausibility": float(report.mean_field_plausibility),
        "scalable_classical_plausibility": float(report.scalable_classical_plausibility),
        "quantum_frontier_pressure": float(report.quantum_frontier_pressure),
        "sign_problem_risk": float(report.sign_problem_risk),
        "stoquasticity": float(report.stoquasticity),
        "factorization_proxy": float(report.factorization_proxy),
        "interaction_pressure": float(report.interaction_pressure),
        "critical_pressure": float(report.critical_pressure),
        "route_prior": report.route_prior,
        "reasons": list(report.reasons),
    }

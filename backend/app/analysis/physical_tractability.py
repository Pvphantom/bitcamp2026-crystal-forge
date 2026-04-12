from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from app.analysis.mf_ansatz_disagreement import MeanFieldAnsatzDisagreementReport
from app.analysis.mf_hysteresis import MeanFieldHysteresisReport
from app.analysis.mf_sensitivity import MeanFieldSensitivityReport
from app.analysis.mf_size_consistency import MeanFieldSizeConsistencyReport
from app.analysis.mf_stability import MeanFieldStabilityReport
from app.domain.problem_spec import ProblemSpec
from app.solvers.base import SolverResult


@dataclass(frozen=True)
class PhysicalTractabilityReport:
    mean_field_plausibility: float
    scalable_classical_plausibility: float
    quantum_frontier_pressure: float
    sign_problem_risk: float
    stoquasticity: float
    factorization_proxy: float
    interaction_pressure: float
    critical_pressure: float
    route_prior: str
    reasons: list[str]


def assess_physical_tractability(
    *,
    problem: ProblemSpec,
    cheap_result: SolverResult,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
) -> PhysicalTractabilityReport:
    if problem.model_family == "hubbard":
        report = _assess_hubbard(
            problem=problem,
            cheap_result=cheap_result,
            stability=stability,
            sensitivity=sensitivity,
            size_consistency=size_consistency,
            ansatz_disagreement=ansatz_disagreement,
            hysteresis=hysteresis,
        )
    elif problem.model_family == "tfim":
        report = _assess_tfim(
            problem=problem,
            cheap_result=cheap_result,
            stability=stability,
            sensitivity=sensitivity,
            size_consistency=size_consistency,
            ansatz_disagreement=ansatz_disagreement,
            hysteresis=hysteresis,
        )
    else:
        raise ValueError(f"Unsupported model family: {problem.model_family}")
    return report


def _assess_hubbard(
    *,
    problem: ProblemSpec,
    cheap_result: SolverResult,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
) -> PhysicalTractabilityReport:
    n_up = cheap_result.site_observables["n_up"]
    n_dn = cheap_result.site_observables["n_dn"]
    d_site = cheap_result.site_observables["D_site"]
    sz_site = cheap_result.site_observables["Sz_site"]

    filling = float(cheap_result.global_observables["n"])
    doping = abs(filling - 1.0)
    avg_coordination = max(1.0, 2.0 * len(cheap_result.bond_observables) / max(problem.nsites, 1))
    interaction_pressure = _clip01(problem.U / max(avg_coordination * problem.t * 3.0, 1e-8))
    half_filling_pressure = _clip01(max(0.0, 0.3 - doping) / 0.3)
    site_spread = max(
        pstdev(n_up) if len(n_up) > 1 else 0.0,
        pstdev(n_dn) if len(n_dn) > 1 else 0.0,
        pstdev(d_site) if len(d_site) > 1 else 0.0,
        pstdev(sz_site) if len(sz_site) > 1 else 0.0,
    )
    correlation_load = max(
        abs(float(cheap_result.global_observables.get("Cs_max", 0.0))) / 0.25,
        abs(float(cheap_result.global_observables.get("Ms2", 0.0))) / 0.35,
    )
    factorization_proxy = _clip01(1.0 - 0.7 * site_spread / 0.2 - 0.3 * min(1.0, correlation_load))
    sign_problem_risk = _clip01((doping / 0.35) * (0.4 + 0.6 * interaction_pressure))
    instability_pressure = _instability_pressure(
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
    )
    quantum_frontier_pressure = _clip01(
        0.45 * interaction_pressure
        + 0.35 * half_filling_pressure
        + 0.20 * instability_pressure
    )
    mean_field_plausibility = _clip01(
        0.45 * factorization_proxy
        + 0.30 * (1.0 - interaction_pressure)
        + 0.20 * _clip01(doping / 0.4)
        + 0.05 * (1.0 - instability_pressure)
    )
    scalable_classical_plausibility = _clip01(
        0.50 * (1.0 - sign_problem_risk)
        + 0.25 * (1.0 - mean_field_plausibility)
        + 0.25 * (1.0 - quantum_frontier_pressure)
    )

    reasons: list[str] = []
    if mean_field_plausibility >= 0.72:
        reasons.append("factorized_weak_or_doped_hubbard")
    if sign_problem_risk >= 0.6:
        reasons.append("fermionic_sign_pressure")
    if quantum_frontier_pressure >= 0.72:
        reasons.append("strong_coupling_half_filling_pressure")
    route_prior = _route_prior(
        mean_field_plausibility=mean_field_plausibility,
        scalable_classical_plausibility=scalable_classical_plausibility,
        quantum_frontier_pressure=quantum_frontier_pressure,
    )
    return PhysicalTractabilityReport(
        mean_field_plausibility=mean_field_plausibility,
        scalable_classical_plausibility=scalable_classical_plausibility,
        quantum_frontier_pressure=quantum_frontier_pressure,
        sign_problem_risk=sign_problem_risk,
        stoquasticity=0.0,
        factorization_proxy=factorization_proxy,
        interaction_pressure=interaction_pressure,
        critical_pressure=half_filling_pressure,
        route_prior=route_prior,
        reasons=reasons,
    )


def _assess_tfim(
    *,
    problem: ProblemSpec,
    cheap_result: SolverResult,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
) -> PhysicalTractabilityReport:
    mz = cheap_result.site_observables["Mz_site"]
    mx = cheap_result.site_observables["Mx_site"]
    field_ratio = abs(problem.h) / max(abs(problem.J), 1e-8)
    bias_ratio = abs(problem.g) / max(abs(problem.J), 1e-8)
    critical_pressure = _clip01(max(0.0, 0.35 - ((field_ratio - 1.0) ** 2 + bias_ratio**2) ** 0.5) / 0.35)
    site_spread = max(
        pstdev(mz) if len(mz) > 1 else 0.0,
        pstdev(mx) if len(mx) > 1 else 0.0,
    )
    correlation_load = max(
        abs(float(cheap_result.global_observables.get("ZZ_nn", 0.0))) / 0.8,
        abs(float(cheap_result.global_observables.get("Mstag2", 0.0))) / 0.35,
    )
    factorization_proxy = _clip01(1.0 - 0.7 * site_spread / 0.25 - 0.3 * min(1.0, correlation_load))
    stoquasticity = 1.0 if problem.J >= 0.0 and problem.h >= 0.0 else 0.0
    sign_problem_risk = 1.0 - stoquasticity
    instability_pressure = _instability_pressure(
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
    )
    quantum_frontier_pressure = _clip01(0.60 * critical_pressure + 0.40 * instability_pressure)
    mean_field_plausibility = _clip01(
        0.45 * factorization_proxy
        + 0.30 * _clip01((field_ratio - 1.0) / 1.5)
        + 0.15 * _clip01(bias_ratio / 0.6)
        + 0.10 * (1.0 - critical_pressure)
    )
    scalable_classical_plausibility = _clip01(
        0.50 * stoquasticity
        + 0.30 * (1.0 - critical_pressure)
        + 0.20 * (1.0 - mean_field_plausibility)
    )

    reasons: list[str] = []
    if stoquasticity >= 1.0:
        reasons.append("stoquastic_tfim")
    if critical_pressure >= 0.7:
        reasons.append("critical_tfim_fluctuations")
    if mean_field_plausibility >= 0.72:
        reasons.append("field_dominated_tfim")
    route_prior = _route_prior(
        mean_field_plausibility=mean_field_plausibility,
        scalable_classical_plausibility=scalable_classical_plausibility,
        quantum_frontier_pressure=quantum_frontier_pressure,
    )
    return PhysicalTractabilityReport(
        mean_field_plausibility=mean_field_plausibility,
        scalable_classical_plausibility=scalable_classical_plausibility,
        quantum_frontier_pressure=quantum_frontier_pressure,
        sign_problem_risk=sign_problem_risk,
        stoquasticity=stoquasticity,
        factorization_proxy=factorization_proxy,
        interaction_pressure=_clip01(abs(problem.J) / max(abs(problem.h) + abs(problem.g), 1e-8)),
        critical_pressure=critical_pressure,
        route_prior=route_prior,
        reasons=reasons,
    )


def _instability_pressure(
    *,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
) -> float:
    return _clip01(
        0.15 * (1.0 - float(stability.converged_fraction))
        + 0.10 * min(1.0, float(stability.energy_span) / 0.1)
        + 0.10 * min(1.0, max(stability.observable_spans.values(), default=0.0) / 0.1)
        + 0.15 * min(1.0, float(sensitivity.observable_shift_max) / 0.15)
        + 0.10 * min(1.0, float(size_consistency.observable_shift_max) / 0.12)
        + 0.20 * min(1.0, float(ansatz_disagreement.max_abs_gap) / 0.12)
        + 0.20 * min(1.0, float(hysteresis.observable_gap_max) / 0.12)
    )


def _route_prior(
    *,
    mean_field_plausibility: float,
    scalable_classical_plausibility: float,
    quantum_frontier_pressure: float,
) -> str:
    if quantum_frontier_pressure >= 0.72 and mean_field_plausibility < 0.7:
        return "quantum_frontier"
    if mean_field_plausibility >= 0.72 and quantum_frontier_pressure < 0.65:
        return "mean_field"
    return "scalable_classical"


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

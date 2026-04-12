from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from app.analysis.mf_ansatz_disagreement import MeanFieldAnsatzDisagreementReport
from app.analysis.mf_hysteresis import MeanFieldHysteresisReport
from app.analysis.mf_sensitivity import MeanFieldSensitivityReport
from app.analysis.mf_size_consistency import MeanFieldSizeConsistencyReport
from app.analysis.mf_stability import MeanFieldStabilityReport
from app.analysis.physical_tractability import PhysicalTractabilityReport
from app.solvers.base import SolverResult


@dataclass(frozen=True)
class GeneralTractabilityFeaturesReport:
    spatial_uniformity: float
    site_dispersion: float
    bond_dispersion: float
    bond_sign_conflict: float
    continuation_curvature: float
    fixed_point_stiffness: float
    metastability_pressure: float
    scale_transfer_stability: float
    response_linearity: float
    correlation_load: float
    classical_obstruction: float
    factorization_reserve: float


def analyze_general_tractability_features(
    *,
    cheap_result: SolverResult,
    stability: MeanFieldStabilityReport,
    sensitivity: MeanFieldSensitivityReport,
    size_consistency: MeanFieldSizeConsistencyReport,
    ansatz_disagreement: MeanFieldAnsatzDisagreementReport,
    hysteresis: MeanFieldHysteresisReport,
    physical_tractability: PhysicalTractabilityReport,
) -> GeneralTractabilityFeaturesReport:
    site_dispersion = _site_dispersion(cheap_result)
    bond_dispersion, bond_sign_conflict, correlation_load = _bond_statistics(cheap_result)
    continuation_curvature = _clip01(
        float(hysteresis.observable_gap_max)
        / max(float(sensitivity.observable_shift_max) + 1e-6, 1e-6)
        / 2.0
    )
    fixed_point_stiffness = _clip01(
        0.45 * float(stability.converged_fraction)
        + 0.35 * (1.0 - min(1.0, float(stability.residual_max) / 1e-4))
        + 0.20 * (1.0 - min(1.0, float(stability.final_delta_mean) / 1e-4))
    )
    metastability_pressure = _clip01(
        0.20 * min(1.0, max(0, stability.distinct_solution_count - 1) / 3.0)
        + 0.25 * min(1.0, float(stability.energy_span) / 0.1)
        + 0.20 * min(1.0, float(ansatz_disagreement.max_abs_gap) / 0.12)
        + 0.20 * min(1.0, float(hysteresis.observable_gap_max) / 0.12)
        + 0.15 * continuation_curvature
    )
    scale_transfer_stability = _clip01(
        1.0
        - 0.65 * min(1.0, float(size_consistency.observable_shift_max) / 0.12)
        - 0.35 * min(1.0, float(size_consistency.energy_density_shift) / 0.08)
    )
    response_linearity = _clip01(
        1.0
        - 0.7 * continuation_curvature
        - 0.3 * min(1.0, float(sensitivity.energy_density_shift_max) / 0.12)
    )
    spatial_uniformity = _clip01(
        1.0
        - 0.55 * min(1.0, site_dispersion / 0.18)
        - 0.25 * min(1.0, bond_dispersion / 0.18)
        - 0.20 * correlation_load
    )
    classical_obstruction = _clip01(
        0.35 * float(physical_tractability.sign_problem_risk)
        + 0.20 * bond_sign_conflict
        + 0.25 * metastability_pressure
        + 0.20 * (1.0 - scale_transfer_stability)
    )
    factorization_reserve = _clip01(
        float(physical_tractability.mean_field_plausibility)
        * spatial_uniformity
        * fixed_point_stiffness
    )
    return GeneralTractabilityFeaturesReport(
        spatial_uniformity=spatial_uniformity,
        site_dispersion=site_dispersion,
        bond_dispersion=bond_dispersion,
        bond_sign_conflict=bond_sign_conflict,
        continuation_curvature=continuation_curvature,
        fixed_point_stiffness=fixed_point_stiffness,
        metastability_pressure=metastability_pressure,
        scale_transfer_stability=scale_transfer_stability,
        response_linearity=response_linearity,
        correlation_load=correlation_load,
        classical_obstruction=classical_obstruction,
        factorization_reserve=factorization_reserve,
    )


def _site_dispersion(cheap_result: SolverResult) -> float:
    spreads: list[float] = []
    for values in cheap_result.site_observables.values():
        if len(values) <= 1:
            continue
        magnitude = max(mean(abs(float(v)) for v in values), 1e-6)
        spreads.append(float(pstdev(float(v) for v in values)) / magnitude)
    return _clip01(max(spreads, default=0.0) / 2.0)


def _bond_statistics(cheap_result: SolverResult) -> tuple[float, float, float]:
    bond_values = [float(value) for value in cheap_result.bond_observables.values()]
    if len(bond_values) <= 1:
        return 0.0, 0.0, 0.0
    bond_mean_mag = max(mean(abs(v) for v in bond_values), 1e-6)
    bond_dispersion = _clip01((float(pstdev(bond_values)) / bond_mean_mag) / 2.0)
    positive = sum(1 for value in bond_values if value > 1e-8)
    negative = sum(1 for value in bond_values if value < -1e-8)
    occupied_signs = int(positive > 0) + int(negative > 0)
    sign_conflict = 0.0 if occupied_signs <= 1 else min(positive, negative) / max(positive + negative, 1)
    correlation_load = _clip01(max(abs(value) for value in bond_values) / 0.5)
    return bond_dispersion, float(sign_conflict), correlation_load


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

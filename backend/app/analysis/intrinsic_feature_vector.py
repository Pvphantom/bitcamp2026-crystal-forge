from __future__ import annotations

import torch


INTRINSIC_AUGMENTED_FEATURE_DIM = 49


def build_intrinsic_augmented_features(sample: dict) -> torch.Tensor:
    base = torch.as_tensor(sample["features"], dtype=torch.float32)
    stability = sample.get("stability", {})
    sensitivity = sample.get("sensitivity", {})
    size_consistency = sample.get("size_consistency", {})
    ansatz = sample.get("ansatz_disagreement", {})
    hysteresis = sample.get("hysteresis", {})
    physical = sample.get("physical_tractability", {})

    extra = torch.tensor(
        [
            float(stability.get("converged_fraction", 0.0)),
            float(stability.get("energy_std", 0.0)),
            float(stability.get("energy_span", 0.0)),
            float(stability.get("residual_mean", 0.0)),
            float(stability.get("residual_max", 0.0)),
            float(stability.get("final_delta_mean", 0.0)),
            float(stability.get("distinct_solution_count", 0.0)),
            float(max((stability.get("observable_spans") or {}).values(), default=0.0)),
            float(sensitivity.get("energy_density_shift_max", 0.0)),
            float(sensitivity.get("observable_shift_max", 0.0)),
            float(max((sensitivity.get("observable_shift_by_param") or {}).values(), default=0.0)),
            float(size_consistency.get("observable_shift_max", 0.0)),
            float(size_consistency.get("energy_density_shift", 0.0)),
            float(max((size_consistency.get("observable_shift_by_name") or {}).values(), default=0.0)),
            float(ansatz.get("max_abs_gap", 0.0)),
            float(ansatz.get("observable_gap_norm", 0.0)),
            float(ansatz.get("energy_density_gap", 0.0)),
            float(hysteresis.get("observable_gap_max", 0.0)),
            float(hysteresis.get("energy_density_gap", 0.0)),
            float(physical.get("mean_field_plausibility", 0.0)),
            float(physical.get("scalable_classical_plausibility", 0.0)),
            float(physical.get("quantum_frontier_pressure", 0.0)),
            float(physical.get("sign_problem_risk", 0.0)),
            float(physical.get("stoquasticity", 0.0)),
            float(physical.get("factorization_proxy", 0.0)),
            float(physical.get("interaction_pressure", 0.0)),
            float(physical.get("critical_pressure", 0.0)),
        ],
        dtype=torch.float32,
    )
    features = torch.cat([base, extra], dim=0)
    if int(features.shape[0]) != INTRINSIC_AUGMENTED_FEATURE_DIM:
        raise ValueError(f"Expected augmented feature dim {INTRINSIC_AUGMENTED_FEATURE_DIM}, got {int(features.shape[0])}")
    return features


def build_runtime_augmented_features(
    base_features: torch.Tensor,
    runtime_report,
) -> torch.Tensor:
    base = torch.as_tensor(base_features, dtype=torch.float32)
    stability = runtime_report.stability
    sensitivity = runtime_report.sensitivity
    size_consistency = runtime_report.size_consistency
    ansatz = runtime_report.ansatz_disagreement
    hysteresis = runtime_report.hysteresis
    physical = runtime_report.physical_tractability
    extra = torch.tensor(
        [
            float(stability.converged_fraction),
            float(stability.energy_std),
            float(stability.energy_span),
            float(stability.residual_mean),
            float(stability.residual_max),
            float(stability.final_delta_mean),
            float(stability.distinct_solution_count),
            float(max(stability.observable_spans.values(), default=0.0)),
            float(sensitivity.energy_density_shift_max),
            float(sensitivity.observable_shift_max),
            float(max(sensitivity.observable_shift_by_param.values(), default=0.0)),
            float(size_consistency.observable_shift_max),
            float(size_consistency.energy_density_shift),
            float(max(size_consistency.observable_shift_by_name.values(), default=0.0)),
            float(ansatz.max_abs_gap),
            float(ansatz.observable_gap_norm),
            float(ansatz.energy_density_gap),
            float(hysteresis.observable_gap_max),
            float(hysteresis.energy_density_gap),
            float(physical.mean_field_plausibility),
            float(physical.scalable_classical_plausibility),
            float(physical.quantum_frontier_pressure),
            float(physical.sign_problem_risk),
            float(physical.stoquasticity),
            float(physical.factorization_proxy),
            float(physical.interaction_pressure),
            float(physical.critical_pressure),
        ],
        dtype=torch.float32,
    )
    features = torch.cat([base, extra], dim=0)
    if int(features.shape[0]) != INTRINSIC_AUGMENTED_FEATURE_DIM:
        raise ValueError(f"Expected augmented feature dim {INTRINSIC_AUGMENTED_FEATURE_DIM}, got {int(features.shape[0])}")
    return features

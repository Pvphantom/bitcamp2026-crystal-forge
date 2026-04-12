from __future__ import annotations

import torch

from app.analysis.intrinsic_feature_vector import (
    INTRINSIC_AUGMENTED_FEATURE_DIM,
    build_intrinsic_augmented_features,
    build_runtime_augmented_features,
)


GENERAL_INTRINSIC_FEATURE_DIM = INTRINSIC_AUGMENTED_FEATURE_DIM + 12


def build_general_intrinsic_features(sample: dict) -> torch.Tensor:
    base = build_intrinsic_augmented_features(sample)
    general = sample.get("general_tractability", {})
    extra = torch.tensor(
        [
            float(general.get("spatial_uniformity", 0.0)),
            float(general.get("site_dispersion", 0.0)),
            float(general.get("bond_dispersion", 0.0)),
            float(general.get("bond_sign_conflict", 0.0)),
            float(general.get("continuation_curvature", 0.0)),
            float(general.get("fixed_point_stiffness", 0.0)),
            float(general.get("metastability_pressure", 0.0)),
            float(general.get("scale_transfer_stability", 0.0)),
            float(general.get("response_linearity", 0.0)),
            float(general.get("correlation_load", 0.0)),
            float(general.get("classical_obstruction", 0.0)),
            float(general.get("factorization_reserve", 0.0)),
        ],
        dtype=torch.float32,
    )
    features = torch.cat([base, extra], dim=0)
    if int(features.shape[0]) != GENERAL_INTRINSIC_FEATURE_DIM:
        raise ValueError(f"Expected general feature dim {GENERAL_INTRINSIC_FEATURE_DIM}, got {int(features.shape[0])}")
    return features


def build_runtime_general_features(base_features: torch.Tensor, runtime_report, general_report) -> torch.Tensor:
    base = build_runtime_augmented_features(base_features, runtime_report)
    extra = torch.tensor(
        [
            float(general_report.spatial_uniformity),
            float(general_report.site_dispersion),
            float(general_report.bond_dispersion),
            float(general_report.bond_sign_conflict),
            float(general_report.continuation_curvature),
            float(general_report.fixed_point_stiffness),
            float(general_report.metastability_pressure),
            float(general_report.scale_transfer_stability),
            float(general_report.response_linearity),
            float(general_report.correlation_load),
            float(general_report.classical_obstruction),
            float(general_report.factorization_reserve),
        ],
        dtype=torch.float32,
    )
    features = torch.cat([base, extra], dim=0)
    if int(features.shape[0]) != GENERAL_INTRINSIC_FEATURE_DIM:
        raise ValueError(f"Expected general feature dim {GENERAL_INTRINSIC_FEATURE_DIM}, got {int(features.shape[0])}")
    return features

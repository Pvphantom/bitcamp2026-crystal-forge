from __future__ import annotations

import torch

from app.ml.qprobe_operator_features import qprobe_operator_feature_dim
from scripts.train_qprobe_general_model import split_by_target_groups


def _sample(family: str, targets: list[str], recommended_cost: int = 1, success: bool = True) -> dict:
    return {
        "features": torch.zeros(qprobe_operator_feature_dim(), dtype=torch.float32),
        "recommended_cost": recommended_cost,
        "full_cost": 6,
        "measurement_savings": 5,
        "success": success,
        "max_abs_error": 0.01,
        "group_bases": ["ZZZZ"],
        "targets": targets,
        "metadata": {
            "family": family,
            "targets": targets,
        },
    }


def test_split_by_target_groups_prevents_combo_leakage() -> None:
    samples = []
    for family in ("hubbard", "tfim"):
        for targets in (["A"], ["B"], ["A", "B"], ["C"]):
            for _ in range(3):
                samples.append(_sample(family, targets))

    train_samples, val_samples, test_samples = split_by_target_groups(samples)
    train_groups = {f"{s['metadata']['family']}|{','.join(sorted(s['metadata']['targets']))}" for s in train_samples}
    val_groups = {f"{s['metadata']['family']}|{','.join(sorted(s['metadata']['targets']))}" for s in val_samples}
    test_groups = {f"{s['metadata']['family']}|{','.join(sorted(s['metadata']['targets']))}" for s in test_samples}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from app.ml.qprobe_adaptive_step_model import AdaptiveStopMLP
from scripts.train_qprobe_adaptive_physics_model import family_group_key, split_by_operator_family, evaluate_with_thresholds


def evaluate(*, dataset_path: Path, model_path: Path) -> dict[str, Any]:
    samples = torch.load(dataset_path, map_location="cpu")
    checkpoint = torch.load(model_path, map_location="cpu")
    family_filter = checkpoint.get("family_filter")
    if family_filter is not None:
        samples = [sample for sample in samples if sample["metadata"]["family"] == family_filter]
    _, _, test_samples = split_by_operator_family(samples)
    model = AdaptiveStopMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"].clamp_min(1e-6)
    stop_prob_threshold = float(checkpoint["stop_prob_threshold"])
    margin_guard = float(checkpoint["margin_guard"])
    test_samples = [{**s, "features": (s["features"] - mean) / std} for s in test_samples]

    from scripts.train_qprobe_adaptive_physics_model import make_loader

    test_loader = make_loader(test_samples, batch_size=128, shuffle=False)
    overall = evaluate_with_thresholds(model, test_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard)

    by_family: dict[str, list[dict]] = {}
    by_op: dict[str, list[dict]] = {}
    for sample in test_samples:
        by_family.setdefault(sample["metadata"]["family"], []).append(sample)
        by_op.setdefault(sample["metadata"]["operator_family"], []).append(sample)

    return {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "split_strategy": "held_out_operator_families",
        "family_filter": family_filter,
        "calibration": {
            "stop_prob_threshold": stop_prob_threshold,
            "margin_guard": margin_guard,
        },
        "overall": overall,
        "by_problem_family": {
            k: evaluate_with_thresholds(model, make_loader(v, batch_size=128, shuffle=False), stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard)
            for k, v in sorted(by_family.items())
        },
        "by_operator_family": {
            k: evaluate_with_thresholds(model, make_loader(v, batch_size=128, shuffle=False), stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard)
            for k, v in sorted(by_op.items())
        },
        "held_out_operator_families": sorted({family_group_key(s) for s in test_samples}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_dataset.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_mlp.pt"))
    parser.add_argument("--out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_eval.json"))
    args = parser.parse_args()
    report = evaluate(dataset_path=args.dataset, model_path=args.model)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

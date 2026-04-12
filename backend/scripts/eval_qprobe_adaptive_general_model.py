from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from app.ml.qprobe_model import QProbeMLP
from scripts.train_qprobe_adaptive_general_model import family_group_key, split_by_operator_family


def _load_model(model_path: Path) -> tuple[QProbeMLP, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = QProbeMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "cost_accuracy": 0.0,
            "success_accuracy": 0.0,
            "error_mae": 0.0,
            "false_safe_rate": 0.0,
            "unsafe_undercompression_rate": 0.0,
        }
    n = len(rows)
    cost_acc = sum(int(r["predicted_cost"] == r["true_cost"]) for r in rows) / n
    success_acc = sum(int(r["predicted_success"] == r["true_success"]) for r in rows) / n
    margin_mae = sum(abs(r["predicted_margin"] - r["true_margin"]) for r in rows) / n
    false_safe_num = sum(int(r["predicted_success"] and not r["true_success"]) for r in rows)
    predicted_safe = sum(int(r["predicted_success"]) for r in rows)
    unsafe_undercompression = sum(int(r["predicted_success"] and r["predicted_cost"] < r["true_cost"]) for r in rows) / n
    return {
        "num_samples": n,
        "cost_accuracy": cost_acc,
        "success_accuracy": success_acc,
        "margin_mae": margin_mae,
        "false_safe_rate": 0.0 if predicted_safe == 0 else false_safe_num / predicted_safe,
        "unsafe_undercompression_rate": unsafe_undercompression,
    }


def evaluate(*, dataset_path: Path, model_path: Path) -> dict[str, Any]:
    samples = torch.load(dataset_path, map_location="cpu")
    _, _, test_samples = split_by_operator_family(samples)
    model, checkpoint = _load_model(model_path)
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"].clamp_min(1e-6)
    cost_classes = [int(x) for x in checkpoint["cost_classes"]]
    success_prob_threshold = float(checkpoint.get("success_prob_threshold", 0.5))
    margin_guard = float(checkpoint.get("margin_guard", float("-inf")))

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in test_samples:
            outputs = model((((sample["features"] - mean) / std).unsqueeze(0)))
            pred_cost = cost_classes[int(torch.argmax(outputs["cost_logits"], dim=-1).item())]
            success_prob = float(torch.softmax(outputs["success_logits"], dim=-1)[0, 1].item())
            pred_margin = float(outputs["error_pred"].item())
            pred_success = success_prob >= success_prob_threshold and pred_margin >= margin_guard
            metadata = sample["metadata"]
            rows.append(
                {
                    "family": metadata["family"],
                    "operator_family": metadata["operator_family"],
                    "num_targets": metadata["num_targets"],
                    "predicted_cost": pred_cost,
                    "true_cost": int(sample["recommended_cost"]),
                    "predicted_success": pred_success,
                    "true_success": bool(sample["success"]),
                    "predicted_margin": pred_margin,
                    "true_margin": float(sample["metadata"]["tolerance"]) - float(sample["max_abs_error"]),
                }
            )

    by_family: dict[str, list[dict[str, Any]]] = {}
    by_op_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
        by_op_family.setdefault(row["operator_family"], []).append(row)
    return {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "split_strategy": "held_out_operator_families",
        "calibration": {
            "success_prob_threshold": success_prob_threshold,
            "margin_guard": margin_guard,
        },
        "overall": _metrics(rows),
        "by_problem_family": {k: _metrics(v) for k, v in sorted(by_family.items())},
        "by_operator_family": {k: _metrics(v) for k, v in sorted(by_op_family.items())},
        "held_out_operator_families": sorted({family_group_key(sample) for sample in test_samples}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_adaptive_synth_dataset.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/qprobe_adaptive_general_mlp.pt"))
    parser.add_argument("--out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_general_eval.json"))
    args = parser.parse_args()
    report = evaluate(dataset_path=args.dataset, model_path=args.model)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

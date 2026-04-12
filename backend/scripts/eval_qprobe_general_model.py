from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from app.ml.qprobe_model import QProbeMLP
from app.ml.schema import (
    DEFAULT_QPROBE_GENERAL_DATASET,
    DEFAULT_QPROBE_GENERAL_METRICS_PATH,
    DEFAULT_QPROBE_GENERAL_MODEL_PATH,
)
from scripts.train_qprobe_general_model import split_by_target_groups, target_group_key


def _load_model(model_path: Path) -> tuple[QProbeMLP, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = QProbeMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _predict_rows(
    *,
    model: QProbeMLP,
    checkpoint: dict[str, Any],
    samples: list[dict],
) -> list[dict[str, Any]]:
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"].clamp_min(1e-6)
    cost_classes = [int(cost) for cost in checkpoint["cost_classes"]]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            normalized = (sample["features"] - feature_mean) / feature_std
            outputs = model(normalized.unsqueeze(0))
            pred_cost_index = int(torch.argmax(outputs["cost_logits"], dim=-1).item())
            pred_success_index = int(torch.argmax(outputs["success_logits"], dim=-1).item())
            pred_error = float(outputs["error_pred"].item())
            metadata = sample["metadata"]
            rows.append(
                {
                    "family": metadata["family"],
                    "target_arity": len(metadata["targets"]),
                    "target_group": target_group_key(sample),
                    "predicted_cost": cost_classes[pred_cost_index],
                    "true_cost": int(sample["recommended_cost"]),
                    "predicted_success": bool(pred_success_index),
                    "true_success": bool(sample["success"]),
                    "predicted_error": pred_error,
                    "true_error": float(sample["max_abs_error"]),
                }
            )
    return rows


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "cost_accuracy": 0.0,
            "success_accuracy": 0.0,
            "error_mae": 0.0,
            "unsafe_undercompression_rate": 0.0,
            "undercompression_rate": 0.0,
            "overconservative_rate": 0.0,
            "false_safe_rate": 0.0,
        }

    num_samples = len(rows)
    cost_correct = sum(int(row["predicted_cost"] == row["true_cost"]) for row in rows)
    success_correct = sum(int(row["predicted_success"] == row["true_success"]) for row in rows)
    error_mae = sum(abs(row["predicted_error"] - row["true_error"]) for row in rows) / num_samples

    undercompression = sum(int(row["predicted_cost"] < row["true_cost"]) for row in rows)
    overconservative = sum(int(row["predicted_cost"] > row["true_cost"]) for row in rows)
    unsafe_undercompression = sum(
        int(row["predicted_success"] and row["predicted_cost"] < row["true_cost"])
        for row in rows
    )
    predicted_safe = sum(int(row["predicted_success"]) for row in rows)
    false_safe = sum(int(row["predicted_success"] and not row["true_success"]) for row in rows)

    return {
        "num_samples": num_samples,
        "cost_accuracy": cost_correct / num_samples,
        "success_accuracy": success_correct / num_samples,
        "error_mae": error_mae,
        "unsafe_undercompression_rate": unsafe_undercompression / num_samples,
        "undercompression_rate": undercompression / num_samples,
        "overconservative_rate": overconservative / num_samples,
        "false_safe_rate": 0.0 if predicted_safe == 0 else false_safe / predicted_safe,
    }


def evaluate(
    *,
    dataset_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    samples = torch.load(dataset_path, map_location="cpu")
    _, _, test_samples = split_by_target_groups(samples)
    model, checkpoint = _load_model(model_path)
    rows = _predict_rows(model=model, checkpoint=checkpoint, samples=test_samples)

    by_family: dict[str, list[dict[str, Any]]] = {}
    by_arity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
        by_arity.setdefault(str(row["target_arity"]), []).append(row)

    return {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "split_strategy": "held_out_target_groups",
        "overall": _metrics_for_rows(rows),
        "by_family": {family: _metrics_for_rows(group_rows) for family, group_rows in sorted(by_family.items())},
        "by_target_arity": {arity: _metrics_for_rows(group_rows) for arity, group_rows in sorted(by_arity.items())},
        "held_out_target_groups": sorted({row["target_group"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_QPROBE_GENERAL_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_QPROBE_GENERAL_MODEL_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_QPROBE_GENERAL_METRICS_PATH.with_name("qprobe_general_eval.json"))
    args = parser.parse_args()

    report = evaluate(dataset_path=args.dataset, model_path=args.model)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

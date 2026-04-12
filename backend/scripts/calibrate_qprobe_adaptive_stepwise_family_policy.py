from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ml.qprobe_adaptive_step_model import AdaptiveStopMLP
from scripts.train_qprobe_adaptive_stepwise_model import (
    evaluate_with_thresholds,
    make_loader,
    split_by_operator_family,
)


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _search_family_thresholds(
    model: AdaptiveStopMLP,
    samples: list[dict],
) -> dict[str, float]:
    loader = make_loader(samples, batch_size=256, shuffle=False)
    probs = np.linspace(0.2, 0.95, 16)
    margins = np.linspace(-0.1, 0.1, 21)
    best = None
    best_metrics = None
    for p in probs:
        for m in margins:
            metrics = evaluate_with_thresholds(model, loader, stop_prob_threshold=float(p), margin_guard=float(m))
            if metrics["false_safe_rate"] <= 0.15 and metrics["predicted_stop_fraction"] >= 0.01:
                candidate = (-metrics["safe_f1"], metrics["false_safe_rate"], -metrics["safe_recall"], p, m)
                if best is None or candidate < best:
                    best = candidate
                    best_metrics = metrics
    if best is None:
        for p in probs:
            for m in margins:
                metrics = evaluate_with_thresholds(model, loader, stop_prob_threshold=float(p), margin_guard=float(m))
                candidate = (metrics["false_safe_rate"], -metrics["safe_f1"], -metrics["safe_recall"], p, m)
                if best is None or candidate < best:
                    best = candidate
                    best_metrics = metrics
    assert best is not None and best_metrics is not None
    *_, p, m = best
    return {
        "stop_prob_threshold": float(p),
        "margin_guard": float(m),
        **best_metrics,
    }


def _aggregate_by_family(
    model: AdaptiveStopMLP,
    samples: list[dict],
    family_policy: dict[str, dict[str, float]],
) -> dict[str, Any]:
    by_family = {}
    total_n = 0
    total_acc = 0.0
    total_mae = 0.0
    total_pred = 0.0
    total_tp = 0.0
    total_true = 0.0
    for family, cfg in family_policy.items():
        family_samples = [sample for sample in samples if sample["metadata"]["family"] == family]
        loader = make_loader(family_samples, batch_size=256, shuffle=False)
        metrics = evaluate_with_thresholds(
            model,
            loader,
            stop_prob_threshold=cfg["stop_prob_threshold"],
            margin_guard=cfg["margin_guard"],
        )
        by_family[family] = metrics
        n = len(family_samples)
        total_n += n
        total_acc += n * metrics["stop_accuracy"]
        total_mae += n * metrics["margin_mae"]
        pred = n * metrics["predicted_stop_fraction"]
        tp = pred * metrics["safe_precision"]
        true = 0.0 if metrics["safe_recall"] == 0 else tp / metrics["safe_recall"]
        total_pred += pred
        total_tp += tp
        total_true += true
    precision = 0.0 if total_pred == 0 else total_tp / total_pred
    recall = 0.0 if total_true == 0 else total_tp / total_true
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    overall = {
        "stop_accuracy": 0.0 if total_n == 0 else total_acc / total_n,
        "margin_mae": 0.0 if total_n == 0 else total_mae / total_n,
        "false_safe_rate": 0.0 if total_pred == 0 else (total_pred - total_tp) / total_pred,
        "predicted_stop_fraction": 0.0 if total_n == 0 else total_pred / total_n,
        "safe_precision": precision,
        "safe_recall": recall,
        "safe_f1": f1,
    }
    return {"overall": overall, "by_family": by_family}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_adaptive_stepwise_dataset_test.pt"))
    parser.add_argument("--model", type=Path, default=Path("backend/artifacts/qprobe_adaptive_stepwise_mlp_smoke.pt"))
    parser.add_argument("--out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_stepwise_family_policy.json"))
    args = parser.parse_args()

    samples = torch.load(args.dataset, map_location="cpu")
    _, val_samples, test_samples = split_by_operator_family(samples)
    checkpoint = torch.load(args.model, map_location="cpu")
    model = AdaptiveStopMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"].clamp_min(1e-6)
    val_samples = _normalize(val_samples, mean, std)
    test_samples = _normalize(test_samples, mean, std)

    families = sorted({sample["metadata"]["family"] for sample in val_samples})
    family_policy = {
        family: _search_family_thresholds(model, [sample for sample in val_samples if sample["metadata"]["family"] == family])
        for family in families
    }
    report = {
        "dataset_path": str(args.dataset),
        "model_path": str(args.model),
        "family_policy": family_policy,
        "test_metrics": _aggregate_by_family(model, test_samples, family_policy),
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

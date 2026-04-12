from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.qprobe_adaptive_step_features import qprobe_adaptive_step_feature_dim
from app.ml.qprobe_adaptive_step_model import AdaptiveStopMLP
from scripts.train_qprobe_adaptive_stepwise_model import family_group_key, split_by_operator_family


def collate(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    stop_labels = torch.tensor([1 if sample["safe_stop"] else 0 for sample in samples], dtype=torch.float32)
    margin_targets = torch.tensor(
        [float(max(-0.2, min(0.05, sample["margin"]))) for sample in samples],
        dtype=torch.float32,
    )
    return features, stop_labels, margin_targets


def make_loader(samples: list[dict], *, batch_size: int, shuffle: bool) -> DataLoader:
    features, stop_labels, margin_targets = collate(samples)
    return DataLoader(TensorDataset(features, stop_labels, margin_targets), batch_size=batch_size, shuffle=shuffle)


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _predict_rows(model: AdaptiveStopMLP, loader: DataLoader) -> list[dict[str, float | bool]]:
    model.eval()
    rows: list[dict[str, float | bool]] = []
    with torch.no_grad():
        for features, stop_y, margin_y in loader:
            outputs = model(features)
            stop_probs = torch.sigmoid(outputs["stop_logit"]).tolist()
            margin_preds = outputs["margin_pred"].tolist()
            for stop_prob, stop_true, margin_pred, margin_true in zip(
                stop_probs, stop_y.tolist(), margin_preds, margin_y.tolist(), strict=True
            ):
                rows.append(
                    {
                        "stop_prob": float(stop_prob),
                        "true_stop": bool(stop_true),
                        "pred_margin": float(max(-0.2, min(0.05, margin_pred))),
                        "true_margin": float(max(-0.2, min(0.05, margin_true))),
                    }
                )
    return rows


def evaluate_with_thresholds(
    model: AdaptiveStopMLP,
    loader: DataLoader,
    *,
    stop_prob_threshold: float,
    margin_guard: float,
) -> dict[str, float]:
    rows = _predict_rows(model, loader)
    preds = [
        1 if row["stop_prob"] >= stop_prob_threshold and row["pred_margin"] >= margin_guard else 0
        for row in rows
    ]
    labels = [1 if row["true_stop"] else 0 for row in rows]
    predicted_stop = sum(preds)
    true_stop = sum(labels)
    false_safe = sum(int(pred == 1 and label == 0) for pred, label in zip(preds, labels, strict=True))
    true_positive = sum(int(pred == 1 and label == 1) for pred, label in zip(preds, labels, strict=True))
    precision = 0.0 if predicted_stop == 0 else true_positive / predicted_stop
    recall = 0.0 if true_stop == 0 else true_positive / true_stop
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "stop_accuracy": accuracy_score(labels, preds),
        "margin_mae": float(
            torch.mean(
                torch.abs(
                    torch.tensor([r["pred_margin"] for r in rows])
                    - torch.tensor([r["true_margin"] for r in rows])
                )
            ).item()
        ),
        "false_safe_rate": 0.0 if predicted_stop == 0 else false_safe / predicted_stop,
        "predicted_stop_fraction": 0.0 if not preds else predicted_stop / len(preds),
        "safe_precision": precision,
        "safe_recall": recall,
        "safe_f1": f1,
        "stop_prob_threshold": stop_prob_threshold,
        "margin_guard": margin_guard,
    }


def calibrate_thresholds(model: AdaptiveStopMLP, loader: DataLoader) -> tuple[float, float, dict[str, float]]:
    rows = _predict_rows(model, loader)
    quantiles = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    prob_values = np.array([r["stop_prob"] for r in rows], dtype=float)
    margin_values = np.array([r["pred_margin"] for r in rows], dtype=float)
    prob_candidates = sorted({round(float(np.quantile(prob_values, q)), 4) for q in quantiles})
    margin_candidates = sorted({round(float(np.quantile(margin_values, q)), 4) for q in quantiles})
    preferred = None
    preferred_metrics = None
    fallback = None
    fallback_metrics = None
    for p in prob_candidates:
        for m in margin_candidates:
            metrics = evaluate_with_thresholds(model, loader, stop_prob_threshold=p, margin_guard=m)
            objective = (
                8.0 * metrics["false_safe_rate"]
                + (1.0 - metrics["safe_f1"])
                + 0.5 * max(0.0, 0.02 - metrics["predicted_stop_fraction"])
            )
            candidate = (objective, metrics["false_safe_rate"], -metrics["safe_f1"], -metrics["safe_precision"], p, m)
            if metrics["false_safe_rate"] <= 0.10 and metrics["predicted_stop_fraction"] >= 0.01:
                if preferred is None or candidate < preferred:
                    preferred = candidate
                    preferred_metrics = metrics
            if fallback is None or candidate < fallback:
                fallback = candidate
                fallback_metrics = metrics
    best = preferred if preferred is not None else fallback
    best_metrics = preferred_metrics if preferred_metrics is not None else fallback_metrics
    assert best is not None and best_metrics is not None
    _, _, _, _, p, m = best
    return float(p), float(m), best_metrics


def _weighted_stop_loss(logits: torch.Tensor, labels: torch.Tensor, *, positive_weight: float, negative_weight: float) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, torch.full_like(labels, positive_weight), torch.full_like(labels, negative_weight))
    return torch.mean(losses * weights)


def train(
    *,
    dataset_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    positive_weight: float = 1.0,
    negative_weight: float = 3.0,
) -> dict[str, object]:
    samples = torch.load(dataset_path, map_location="cpu")
    train_samples, val_samples, test_samples = split_by_operator_family(samples)

    train_features, _, _ = collate(train_samples)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    train_samples = _normalize(train_samples, feature_mean, feature_std)
    val_samples = _normalize(val_samples, feature_mean, feature_std)
    test_samples = _normalize(test_samples, feature_mean, feature_std)

    train_loader = make_loader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_samples, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_samples, batch_size=batch_size, shuffle=False)

    model = AdaptiveStopMLP(input_dim=qprobe_adaptive_step_feature_dim())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    margin_loss = nn.SmoothL1Loss(beta=0.05)

    best_state = None
    best_val = float("inf")
    stale = 0
    best_thresholds: tuple[float, float] | None = None
    best_calibration: dict[str, float] | None = None
    for _ in range(epochs):
        model.train()
        for features, stop_y, margin_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = (
                1.5 * _weighted_stop_loss(
                    outputs["stop_logit"],
                    stop_y,
                    positive_weight=positive_weight,
                    negative_weight=negative_weight,
                )
                + 1.0 * margin_loss(outputs["margin_pred"], margin_y)
            )
            loss.backward()
            optimizer.step()
        stop_prob_threshold, margin_guard, calibration = calibrate_thresholds(model, val_loader)
        val_metrics = evaluate_with_thresholds(
            model,
            val_loader,
            stop_prob_threshold=stop_prob_threshold,
            margin_guard=margin_guard,
        )
        val_score = (
            8.0 * val_metrics["false_safe_rate"]
            + (1.0 - val_metrics["safe_f1"])
            + 0.5 * max(0.0, 0.02 - val_metrics["predicted_stop_fraction"])
            + val_metrics["margin_mae"]
        )
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thresholds = (stop_prob_threshold, margin_guard)
            best_calibration = calibration
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break

    assert best_state is not None and best_thresholds is not None and best_calibration is not None
    model.load_state_dict(best_state)
    stop_prob_threshold, margin_guard = best_thresholds
    metrics = {
        "train": evaluate_with_thresholds(model, train_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "val": evaluate_with_thresholds(model, val_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "test": evaluate_with_thresholds(model, test_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "split_strategy": "held_out_operator_families",
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "train_operator_families": sorted({family_group_key(s) for s in train_samples}),
        "val_operator_families": sorted({family_group_key(s) for s in val_samples}),
        "test_operator_families": sorted({family_group_key(s) for s in test_samples}),
        "calibration": best_calibration,
    }

    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {"input_dim": qprobe_adaptive_step_feature_dim(), "hidden_dim": 64},
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "stop_prob_threshold": stop_prob_threshold,
            "margin_guard": margin_guard,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=3.0)
    args = parser.parse_args()
    metrics = train(
        dataset_path=args.dataset,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positive_weight=args.positive_weight,
        negative_weight=args.negative_weight,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

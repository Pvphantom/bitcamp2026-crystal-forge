from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.qprobe_adaptive_physics_features import qprobe_adaptive_physics_feature_dim
from app.ml.qprobe_adaptive_step_model import AdaptiveStopMLP


def collate(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    coverage_labels = torch.tensor([1 if sample["coverage_complete"] else 0 for sample in samples], dtype=torch.float32)
    stop_labels = torch.tensor([1 if sample["safe_stop"] else 0 for sample in samples], dtype=torch.float32)
    margin_targets = torch.tensor(
        [float(max(-0.2, min(0.05, sample["margin"]))) for sample in samples],
        dtype=torch.float32,
    )
    return features, coverage_labels, stop_labels, margin_targets


def make_loader(samples: list[dict], *, batch_size: int, shuffle: bool) -> DataLoader:
    features, coverage_labels, stop_labels, margin_targets = collate(samples)
    return DataLoader(TensorDataset(features, coverage_labels, stop_labels, margin_targets), batch_size=batch_size, shuffle=shuffle)


def family_group_key(sample: dict) -> str:
    m = sample["metadata"]
    return f"{m['family']}|{m['operator_family']}"


def split_by_operator_family(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups = [family_group_key(sample) for sample in samples]
    indices = list(range(len(samples)))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))
    train_val_samples = [samples[i] for i in train_val_idx]
    test_samples = [samples[i] for i in test_idx]

    inner_groups = [family_group_key(sample) for sample in train_val_samples]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=23)
    train_idx, val_idx = next(inner.split(list(range(len(train_val_samples))), groups=inner_groups))
    train_samples = [train_val_samples[i] for i in train_idx]
    val_samples = [train_val_samples[i] for i in val_idx]
    return train_samples, val_samples, test_samples


def _predict_rows(model: AdaptiveStopMLP, loader: DataLoader) -> list[dict[str, float | bool]]:
    model.eval()
    rows: list[dict[str, float | bool]] = []
    with torch.no_grad():
        for features, coverage_y, stop_y, margin_y in loader:
            outputs = model(features)
            stop_probs = torch.sigmoid(outputs["stop_logit"]).tolist()
            margin_preds = outputs["margin_pred"].tolist()
            for stop_prob, coverage_true, stop_true, margin_pred, margin_true in zip(
                stop_probs, coverage_y.tolist(), stop_y.tolist(), margin_preds, margin_y.tolist(), strict=True
            ):
                rows.append(
                    {
                        "stop_prob": float(stop_prob),
                        "coverage_complete": bool(coverage_true),
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
    quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prob_values = np.array([r["stop_prob"] for r in rows], dtype=float)
    margin_values = np.array([r["pred_margin"] for r in rows], dtype=float)
    prob_candidates = sorted({round(float(np.quantile(prob_values, q)), 4) for q in quantiles})
    margin_candidates = sorted({round(float(np.quantile(margin_values, q)), 4) for q in quantiles})
    true_stop_fraction = sum(int(r["true_stop"]) for r in rows) / len(rows)
    preferred = None
    preferred_metrics = None
    fallback = None
    fallback_metrics = None
    target_fraction = min(0.25 * true_stop_fraction, 0.06)
    for p in prob_candidates:
        for m in margin_candidates:
            metrics = evaluate_with_thresholds(model, loader, stop_prob_threshold=p, margin_guard=m)
            objective = (
                5.0 * metrics["false_safe_rate"]
                + (1.0 - metrics["safe_f1"])
                + 0.5 * max(0.0, target_fraction - metrics["predicted_stop_fraction"])
            )
            candidate = (objective, metrics["false_safe_rate"], -metrics["safe_f1"], -metrics["safe_recall"], p, m)
            if metrics["false_safe_rate"] <= 0.15 and metrics["predicted_stop_fraction"] >= target_fraction:
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


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def calibrate_thresholds_by_family(model: AdaptiveStopMLP, samples: list[dict]) -> dict[str, dict[str, float]]:
    families = sorted({sample["metadata"]["family"] for sample in samples})
    out: dict[str, dict[str, float]] = {}
    for family in families:
        family_samples = [sample for sample in samples if sample["metadata"]["family"] == family]
        loader = make_loader(family_samples, batch_size=256, shuffle=False)
        stop_prob_threshold, margin_guard, calibration = calibrate_thresholds(model, loader)
        out[family] = {
            "stop_prob_threshold": stop_prob_threshold,
            "margin_guard": margin_guard,
            **calibration,
        }
    return out


def evaluate_family_policy(
    model: AdaptiveStopMLP,
    samples: list[dict],
    family_calibration: dict[str, dict[str, float]],
) -> dict[str, object]:
    by_family = {}
    total_n = 0
    total_acc = 0.0
    total_mae = 0.0
    total_pred = 0.0
    total_tp = 0.0
    total_true = 0.0
    for family, cfg in family_calibration.items():
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


def train(
    *,
    dataset_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    family_filter: str | None = None,
    stop_pos_weight: float = 3.0,
) -> dict[str, object]:
    samples = torch.load(dataset_path, map_location="cpu")
    if family_filter is not None:
        samples = [sample for sample in samples if sample["metadata"]["family"] == family_filter]
    train_samples, val_samples, test_samples = split_by_operator_family(samples)

    train_features, _, _, _ = collate(train_samples)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    train_samples = _normalize(train_samples, feature_mean, feature_std)
    val_samples = _normalize(val_samples, feature_mean, feature_std)
    test_samples = _normalize(test_samples, feature_mean, feature_std)

    train_loader = make_loader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_samples, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_samples, batch_size=batch_size, shuffle=False)

    model = AdaptiveStopMLP(input_dim=qprobe_adaptive_physics_feature_dim())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    stop_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(stop_pos_weight))
    margin_loss = nn.SmoothL1Loss(beta=0.05)

    best_state = None
    best_val = float("inf")
    stale = 0
    patience = 20
    best_thresholds: tuple[float, float] | None = None
    best_calibration: dict[str, float] | None = None
    for _ in range(epochs):
        model.train()
        for features, coverage_y, stop_y, margin_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = (
                1.2 * stop_loss(outputs["stop_logit"], coverage_y)
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
            5.0 * val_metrics["false_safe_rate"]
            + (1.0 - val_metrics["safe_f1"])
            + 0.5 * max(
                0.0,
                min(0.5 * calibration["predicted_stop_fraction"] + 0.02, 0.15)
                - val_metrics["predicted_stop_fraction"],
            )
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
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    assert best_thresholds is not None and best_calibration is not None
    stop_prob_threshold, margin_guard = best_thresholds
    calibration = best_calibration
    family_calibration = calibrate_thresholds_by_family(model, val_samples)
    family_policy_train = evaluate_family_policy(model, train_samples, family_calibration)
    family_policy_val = evaluate_family_policy(model, val_samples, family_calibration)
    family_policy_test = evaluate_family_policy(model, test_samples, family_calibration)
    metrics = {
        "train": evaluate_with_thresholds(model, train_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "val": evaluate_with_thresholds(model, val_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "test": evaluate_with_thresholds(model, test_loader, stop_prob_threshold=stop_prob_threshold, margin_guard=margin_guard),
        "family_policy_train": family_policy_train["overall"],
        "family_policy_val": family_policy_val["overall"],
        "family_policy_test": family_policy_test["overall"],
        "family_policy_by_family_test": family_policy_test["by_family"],
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "split_strategy": "held_out_operator_families",
        "family_filter": family_filter,
        "stop_pos_weight": stop_pos_weight,
        "train_operator_families": sorted({family_group_key(s) for s in train_samples}),
        "val_operator_families": sorted({family_group_key(s) for s in val_samples}),
        "test_operator_families": sorted({family_group_key(s) for s in test_samples}),
        "calibration": calibration,
        "family_calibration": family_calibration,
    }

    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {"input_dim": qprobe_adaptive_physics_feature_dim(), "hidden_dim": 64},
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "stop_prob_threshold": stop_prob_threshold,
            "margin_guard": margin_guard,
            "family_calibration": family_calibration,
            "family_filter": family_filter,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_dataset.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_mlp.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_physics_metrics.json"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--family-filter", type=str, default=None)
    parser.add_argument("--stop-pos-weight", type=float, default=3.0)
    args = parser.parse_args()
    metrics = train(
        dataset_path=args.dataset,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        family_filter=args.family_filter,
        stop_pos_weight=args.stop_pos_weight,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

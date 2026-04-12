from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.qprobe_model import QProbeMLP
from app.ml.qprobe_adaptive_features import qprobe_adaptive_feature_dim


def collate(samples: list[dict], *, cost_to_index: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    cost_labels = torch.tensor([cost_to_index[int(sample["recommended_cost"])] for sample in samples], dtype=torch.long)
    success_labels = torch.tensor([1 if sample["success"] else 0 for sample in samples], dtype=torch.long)
    margin_targets = torch.tensor(
        [float(sample["metadata"]["tolerance"]) - float(sample["max_abs_error"]) for sample in samples],
        dtype=torch.float32,
    )
    return features, cost_labels, success_labels, margin_targets


def make_loader(
    samples: list[dict],
    *,
    cost_to_index: dict[int, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features, cost_labels, success_labels, margin_targets = collate(samples, cost_to_index=cost_to_index)
    return DataLoader(TensorDataset(features, cost_labels, success_labels, margin_targets), batch_size=batch_size, shuffle=shuffle)


def family_group_key(sample: dict) -> str:
    metadata = sample["metadata"]
    return f"{metadata['family']}|{metadata['operator_family']}"


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


def evaluate(model: QProbeMLP, loader: DataLoader, *, cost_classes: list[int]) -> dict[str, object]:
    return evaluate_with_thresholds(model, loader, cost_classes=cost_classes, success_prob_threshold=0.5, margin_guard=float("-inf"))


def _predict_batches(model: QProbeMLP, loader: DataLoader, *, cost_classes: list[int]) -> list[dict[str, float | int | bool]]:
    model.eval()
    rows: list[dict[str, float | int | bool]] = []
    with torch.no_grad():
        for features, cost_y, success_y, margin_y in loader:
            outputs = model(features)
            cost_pred_indices = torch.argmax(outputs["cost_logits"], dim=-1).tolist()
            success_probs = torch.softmax(outputs["success_logits"], dim=-1)[:, 1].tolist()
            margin_preds = outputs["error_pred"].tolist()
            for cost_index, cost_true, success_prob, success_true, margin_pred, margin_true in zip(
                cost_pred_indices,
                cost_y.tolist(),
                success_probs,
                success_y.tolist(),
                margin_preds,
                margin_y.tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "predicted_cost": cost_classes[cost_index],
                        "true_cost": cost_classes[cost_true],
                        "success_prob": float(success_prob),
                        "true_success": bool(success_true),
                        "predicted_margin": float(margin_pred),
                        "true_margin": float(margin_true),
                    }
                )
    return rows


def evaluate_with_thresholds(
    model: QProbeMLP,
    loader: DataLoader,
    *,
    cost_classes: list[int],
    success_prob_threshold: float,
    margin_guard: float,
) -> dict[str, object]:
    rows = _predict_batches(model, loader, cost_classes=cost_classes)
    cost_preds = [cost_classes.index(int(row["predicted_cost"])) for row in rows]
    cost_labels = [cost_classes.index(int(row["true_cost"])) for row in rows]
    success_preds = [
        1 if float(row["success_prob"]) >= success_prob_threshold and float(row["predicted_margin"]) >= margin_guard else 0
        for row in rows
    ]
    success_labels = [1 if bool(row["true_success"]) else 0 for row in rows]
    margin_preds = [float(row["predicted_margin"]) for row in rows]
    margin_targets = [float(row["true_margin"]) for row in rows]

    false_safe = 0
    predicted_safe = 0
    true_safe = sum(success_labels)
    true_positive = 0
    for pred_success, true_success in zip(success_preds, success_labels, strict=True):
        if pred_success == 1:
            predicted_safe += 1
            if true_success == 1:
                true_positive += 1
            else:
                false_safe += 1
    safe_precision = 0.0 if predicted_safe == 0 else true_positive / predicted_safe
    safe_recall = 0.0 if true_safe == 0 else true_positive / true_safe
    safe_f1 = 0.0 if safe_precision + safe_recall == 0 else 2.0 * safe_precision * safe_recall / (safe_precision + safe_recall)
    return {
        "cost_accuracy": accuracy_score(cost_labels, cost_preds),
        "success_accuracy": accuracy_score(success_labels, success_preds),
        "margin_mae": float(torch.mean(torch.abs(torch.tensor(margin_preds) - torch.tensor(margin_targets))).item()),
        "cost_confusion_matrix": confusion_matrix(cost_labels, cost_preds, labels=list(range(len(cost_classes)))).tolist(),
        "false_safe_rate": 0.0 if predicted_safe == 0 else false_safe / predicted_safe,
        "predicted_safe_fraction": 0.0 if not success_preds else sum(success_preds) / len(success_preds),
        "safe_precision": safe_precision,
        "safe_recall": safe_recall,
        "safe_f1": safe_f1,
        "success_prob_threshold": success_prob_threshold,
        "margin_guard": margin_guard,
    }


def calibrate_thresholds(model: QProbeMLP, loader: DataLoader, *, cost_classes: list[int]) -> tuple[float, float, dict[str, object]]:
    rows = _predict_batches(model, loader, cost_classes=cost_classes)
    prob_candidates = sorted({round(float(row["success_prob"]), 4) for row in rows})
    margin_candidates = sorted({round(float(row["predicted_margin"]), 4) for row in rows})
    true_safe_fraction = sum(int(bool(row["true_success"])) for row in rows) / len(rows)
    preferred = None
    preferred_metrics: dict[str, object] | None = None
    fallback = None
    fallback_metrics: dict[str, object] | None = None
    for prob_threshold in prob_candidates:
        for margin_guard in margin_candidates:
            success_preds = [
                1 if float(row["success_prob"]) >= prob_threshold and float(row["predicted_margin"]) >= margin_guard else 0
                for row in rows
            ]
            success_labels = [1 if bool(row["true_success"]) else 0 for row in rows]
            predicted_safe = sum(success_preds)
            false_safe_num = sum(
                int(pred_success == 1 and true_success == 0)
                for pred_success, true_success in zip(success_preds, success_labels, strict=True)
            )
            true_positive = sum(
                int(pred_success == 1 and true_success == 1)
                for pred_success, true_success in zip(success_preds, success_labels, strict=True)
            )
            true_safe = sum(success_labels)
            false_safe_rate = 0.0 if predicted_safe == 0 else false_safe_num / predicted_safe
            success_accuracy = accuracy_score(success_labels, success_preds)
            predicted_safe_fraction = 0.0 if not success_preds else predicted_safe / len(success_preds)
            safe_precision = 0.0 if predicted_safe == 0 else true_positive / predicted_safe
            safe_recall = 0.0 if true_safe == 0 else true_positive / true_safe
            safe_f1 = 0.0 if safe_precision + safe_recall == 0 else 2.0 * safe_precision * safe_recall / (safe_precision + safe_recall)
            objective = (
                4.0 * false_safe_rate
                + (1.0 - safe_f1)
                + 0.5 * max(0.0, min(0.5 * true_safe_fraction, 0.12) - predicted_safe_fraction)
            )
            candidate = (
                objective,
                false_safe_rate,
                -safe_f1,
                -safe_recall,
                prob_threshold,
                margin_guard,
            )
            metrics = {
                "false_safe_rate": false_safe_rate,
                "success_accuracy": success_accuracy,
                "predicted_safe_fraction": predicted_safe_fraction,
                "true_safe_fraction": true_safe_fraction,
                "safe_precision": safe_precision,
                "safe_recall": safe_recall,
                "safe_f1": safe_f1,
            }
            if false_safe_rate <= 0.15 and predicted_safe_fraction >= min(0.5 * true_safe_fraction, 0.12):
                if preferred is None or candidate < preferred:
                    preferred = candidate
                    preferred_metrics = metrics
            if fallback is None or candidate < fallback:
                fallback = candidate
                fallback_metrics = metrics
    best = preferred if preferred is not None else fallback
    best_metrics = preferred_metrics if preferred_metrics is not None else fallback_metrics
    assert best is not None and best_metrics is not None
    _, _, _, _, prob_threshold, margin_guard = best
    return float(prob_threshold), float(margin_guard), best_metrics


def _normalize(samples: list[dict], feature_mean: torch.Tensor, feature_std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - feature_mean) / feature_std} for sample in samples]


def train(
    *,
    dataset_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    samples = torch.load(dataset_path, map_location="cpu")
    cost_classes = sorted({int(sample["recommended_cost"]) for sample in samples})
    cost_to_index = {cost: index for index, cost in enumerate(cost_classes)}
    train_samples, val_samples, test_samples = split_by_operator_family(samples)

    train_features, _, _, _ = collate(train_samples, cost_to_index=cost_to_index)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    train_samples = _normalize(train_samples, feature_mean, feature_std)
    val_samples = _normalize(val_samples, feature_mean, feature_std)
    test_samples = _normalize(test_samples, feature_mean, feature_std)

    train_loader = make_loader(train_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=False)

    cost_counts = Counter(int(sample["recommended_cost"]) for sample in train_samples)
    cost_weights = torch.tensor([1.0 / cost_counts.get(cost, 1) for cost in cost_classes], dtype=torch.float32)
    cost_weights = cost_weights / cost_weights.sum() * len(cost_classes)

    model = QProbeMLP(input_dim=qprobe_adaptive_feature_dim(), num_cost_classes=len(cost_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cost_loss = nn.CrossEntropyLoss(weight=cost_weights)
    success_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5], dtype=torch.float32))
    error_loss = nn.L1Loss()

    best_state = None
    best_val = float("inf")
    stale = 0
    patience = 25
    for _ in range(epochs):
        model.train()
        for features, cost_y, success_y, margin_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = (
                0.8 * cost_loss(outputs["cost_logits"], cost_y)
                + 1.2 * success_loss(outputs["success_logits"], success_y)
                + 0.8 * error_loss(outputs["error_pred"], margin_y)
            )
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, cost_classes=cost_classes)
        val_score = (
            (1.0 - metrics["cost_accuracy"])
            + 1.5 * metrics["false_safe_rate"]
            + 0.5 * (1.0 - metrics["success_accuracy"])
            + metrics["margin_mae"]
        )
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    success_prob_threshold, margin_guard, calibration_metrics = calibrate_thresholds(model, val_loader, cost_classes=cost_classes)
    metrics = {
        "train": evaluate_with_thresholds(
            model,
            train_loader,
            cost_classes=cost_classes,
            success_prob_threshold=success_prob_threshold,
            margin_guard=margin_guard,
        ),
        "val": evaluate_with_thresholds(
            model,
            val_loader,
            cost_classes=cost_classes,
            success_prob_threshold=success_prob_threshold,
            margin_guard=margin_guard,
        ),
        "test": evaluate_with_thresholds(
            model,
            test_loader,
            cost_classes=cost_classes,
            success_prob_threshold=success_prob_threshold,
            margin_guard=margin_guard,
        ),
        "cost_classes": cost_classes,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "split_strategy": "held_out_operator_families",
        "train_operator_families": sorted({family_group_key(sample) for sample in train_samples}),
        "val_operator_families": sorted({family_group_key(sample) for sample in val_samples}),
        "test_operator_families": sorted({family_group_key(sample) for sample in test_samples}),
        "calibration": {
            "success_prob_threshold": success_prob_threshold,
            "margin_guard": margin_guard,
            **calibration_metrics,
        },
    }
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {"input_dim": qprobe_adaptive_feature_dim(), "hidden_dim": 64, "num_cost_classes": len(cost_classes)},
            "cost_classes": cost_classes,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "success_prob_threshold": success_prob_threshold,
            "margin_guard": margin_guard,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_adaptive_synth_dataset.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_general_mlp.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/qprobe_adaptive_general_metrics.json"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    metrics = train(
        dataset_path=args.dataset,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

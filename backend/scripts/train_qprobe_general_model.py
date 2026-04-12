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
from app.ml.qprobe_operator_features import qprobe_operator_feature_dim
from app.ml.schema import (
    ARTIFACTS_DIR,
    DEFAULT_QPROBE_GENERAL_DATASET,
    DEFAULT_QPROBE_GENERAL_METRICS_PATH,
    DEFAULT_QPROBE_GENERAL_MODEL_PATH,
)


def collate(
    samples: list[dict],
    *,
    cost_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    cost_labels = torch.tensor([cost_to_index[sample["recommended_cost"]] for sample in samples], dtype=torch.long)
    success_labels = torch.tensor([1 if sample["success"] else 0 for sample in samples], dtype=torch.long)
    error_targets = torch.tensor([float(sample["max_abs_error"]) for sample in samples], dtype=torch.float32)
    return features, cost_labels, success_labels, error_targets


def make_loader(
    samples: list[dict],
    *,
    cost_to_index: dict[int, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features, cost_labels, success_labels, error_targets = collate(samples, cost_to_index=cost_to_index)
    dataset = TensorDataset(features, cost_labels, success_labels, error_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def target_group_key(sample: dict) -> str:
    metadata = sample["metadata"]
    return f"{metadata['family']}|{','.join(sorted(metadata['targets']))}"


def split_by_target_groups(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    indices = list(range(len(samples)))
    groups = [target_group_key(sample) for sample in samples]

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))
    train_val_samples = [samples[i] for i in train_val_idx]
    test_samples = [samples[i] for i in test_idx]

    inner_groups = [target_group_key(sample) for sample in train_val_samples]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
    train_idx, val_idx = next(inner.split(list(range(len(train_val_samples))), groups=inner_groups))
    train_samples = [train_val_samples[i] for i in train_idx]
    val_samples = [train_val_samples[i] for i in val_idx]
    return train_samples, val_samples, test_samples


def evaluate(model: QProbeMLP, loader: DataLoader, *, cost_classes: list[int]) -> dict[str, object]:
    model.eval()
    cost_preds: list[int] = []
    cost_labels: list[int] = []
    success_preds: list[int] = []
    success_labels: list[int] = []
    error_preds: list[float] = []
    error_targets: list[float] = []
    with torch.no_grad():
        for features, cost_y, success_y, error_y in loader:
            outputs = model(features)
            cost_preds.extend(torch.argmax(outputs["cost_logits"], dim=-1).tolist())
            cost_labels.extend(cost_y.tolist())
            success_preds.extend(torch.argmax(outputs["success_logits"], dim=-1).tolist())
            success_labels.extend(success_y.tolist())
            error_preds.extend(outputs["error_pred"].tolist())
            error_targets.extend(error_y.tolist())
    cost_acc = accuracy_score(cost_labels, cost_preds)
    success_acc = accuracy_score(success_labels, success_preds)
    mae = float(torch.mean(torch.abs(torch.tensor(error_preds) - torch.tensor(error_targets))).item())

    false_safe = 0
    unsafe_recommendations = 0
    for pred_cost_index, pred_success, true_success in zip(cost_preds, success_preds, success_labels, strict=True):
        predicted_cost = cost_classes[pred_cost_index]
        if pred_success == 1:
            unsafe_recommendations += 1
            if true_success == 0:
                false_safe += 1
    false_safe_rate = 0.0 if unsafe_recommendations == 0 else false_safe / unsafe_recommendations

    return {
        "cost_accuracy": cost_acc,
        "success_accuracy": success_acc,
        "error_mae": mae,
        "cost_confusion_matrix": confusion_matrix(cost_labels, cost_preds, labels=list(range(len(cost_classes)))).tolist(),
        "false_safe_rate": false_safe_rate,
    }


def _normalize(samples: list[dict], feature_mean: torch.Tensor, feature_std: torch.Tensor) -> list[dict]:
    normalized = []
    for sample in samples:
        normalized.append({**sample, "features": (sample["features"] - feature_mean) / feature_std})
    return normalized


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
    train_samples, val_samples, test_samples = split_by_target_groups(samples)

    train_features, _, _, _ = collate(train_samples, cost_to_index=cost_to_index)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    train_samples = _normalize(train_samples, feature_mean, feature_std)
    val_samples = _normalize(val_samples, feature_mean, feature_std)
    test_samples = _normalize(test_samples, feature_mean, feature_std)

    train_loader = make_loader(train_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_samples, cost_to_index=cost_to_index, batch_size=batch_size, shuffle=False)

    cost_counts = Counter(sample["recommended_cost"] for sample in train_samples)
    cost_weights = torch.tensor(
        [1.0 / cost_counts.get(cost, 1) for cost in cost_classes],
        dtype=torch.float32,
    )
    cost_weights = cost_weights / cost_weights.sum() * len(cost_classes)

    model = QProbeMLP(input_dim=qprobe_operator_feature_dim(), num_cost_classes=len(cost_classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cost_loss = nn.CrossEntropyLoss(weight=cost_weights)
    success_loss = nn.CrossEntropyLoss()
    error_loss = nn.L1Loss()

    best_state = None
    best_val = float("inf")
    patience = 25
    stale = 0

    for _ in range(epochs):
        model.train()
        for features, cost_y, success_y, error_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = (
                cost_loss(outputs["cost_logits"], cost_y)
                + 0.5 * success_loss(outputs["success_logits"], success_y)
                + 0.5 * error_loss(outputs["error_pred"], error_y)
            )
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, cost_classes=cost_classes)
        val_score = (1.0 - metrics["cost_accuracy"]) + 0.5 * (1.0 - metrics["success_accuracy"]) + metrics["error_mae"]
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, cost_classes=cost_classes)
    val_metrics = evaluate(model, val_loader, cost_classes=cost_classes)
    test_metrics = evaluate(model, test_loader, cost_classes=cost_classes)
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "cost_classes": cost_classes,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "split_strategy": "held_out_target_groups",
        "train_target_groups": sorted({target_group_key(sample) for sample in train_samples}),
        "val_target_groups": sorted({target_group_key(sample) for sample in val_samples}),
        "test_target_groups": sorted({target_group_key(sample) for sample in test_samples}),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": {
                "input_dim": qprobe_operator_feature_dim(),
                "hidden_dim": 64,
                "num_cost_classes": len(cost_classes),
            },
            "cost_classes": cost_classes,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_QPROBE_GENERAL_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_QPROBE_GENERAL_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_QPROBE_GENERAL_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
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

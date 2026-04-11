from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.schema import (
    ARTIFACTS_DIR,
    DEFAULT_TRUST_DATASET,
    DEFAULT_TRUST_METRICS_PATH,
    DEFAULT_TRUST_MODEL_PATH,
    TRUST_LABELS,
    TRUST_TO_INDEX,
)
from app.ml.trust_model import TrustMLP


def collate(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    risk_labels = torch.tensor([TRUST_TO_INDEX[sample["risk_label"]] for sample in samples], dtype=torch.long)
    error_targets = torch.tensor([float(sample["max_abs_error"]) for sample in samples], dtype=torch.float32)
    return features, risk_labels, error_targets


def make_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features, risk_labels, error_targets = collate(samples)
    dataset = TensorDataset(features, risk_labels, error_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def stratify_keys(samples: list[dict]) -> list[str]:
    return [sample["risk_label"] for sample in samples]


def evaluate(
    model: TrustMLP,
    loader: DataLoader,
    *,
    safe_error_guard: float | None = None,
    safe_prob_guard: float | None = None,
) -> dict[str, object]:
    model.eval()
    pred_labels = []
    true_labels = []
    pred_errors = []
    true_errors = []
    with torch.no_grad():
        for features, risk_y, error_y in loader:
            outputs = model(features)
            logits = outputs["risk_logits"]
            probs = torch.softmax(logits, dim=-1)
            batch_preds = torch.argmax(logits, dim=-1)
            if safe_error_guard is not None or safe_prob_guard is not None:
                safe_index = TRUST_TO_INDEX["safe"]
                warning_index = TRUST_TO_INDEX["warning"]
                downgrade_mask = batch_preds == safe_index
                if safe_error_guard is not None:
                    downgrade_mask = downgrade_mask & (outputs["error_pred"] > safe_error_guard)
                if safe_prob_guard is not None:
                    downgrade_mask = downgrade_mask | (
                        (batch_preds == safe_index) & (probs[:, safe_index] < safe_prob_guard)
                    )
                batch_preds = torch.where(
                    downgrade_mask,
                    torch.full_like(batch_preds, warning_index),
                    batch_preds,
                )
            pred_labels.extend(batch_preds.tolist())
            true_labels.extend(risk_y.tolist())
            pred_errors.extend(outputs["error_pred"].tolist())
            true_errors.extend(error_y.tolist())
    acc = accuracy_score(true_labels, pred_labels)
    mae = float(torch.mean(torch.abs(torch.tensor(pred_errors) - torch.tensor(true_errors))).item())
    false_safe_numer = 0
    false_safe_denom = 0
    for pred_idx, true_idx in zip(pred_labels, true_labels, strict=True):
        if pred_idx == TRUST_TO_INDEX["safe"]:
            false_safe_denom += 1
            if true_idx != TRUST_TO_INDEX["safe"]:
                false_safe_numer += 1
    false_safe_rate = 0.0 if false_safe_denom == 0 else false_safe_numer / false_safe_denom
    return {
        "risk_accuracy": acc,
        "error_mae": mae,
        "false_safe_rate": false_safe_rate,
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=list(range(len(TRUST_LABELS)))).tolist(),
    }


def select_safe_guards(model: TrustMLP, loader: DataLoader) -> tuple[float, float]:
    error_candidates = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    prob_candidates = [0.4, 0.5, 0.6, 0.7, 0.8]
    best_error = error_candidates[-1]
    best_prob = prob_candidates[0]
    best_score = float("inf")
    for error_guard in error_candidates:
        for prob_guard in prob_candidates:
            metrics = evaluate(
                model,
                loader,
                safe_error_guard=error_guard,
                safe_prob_guard=prob_guard,
            )
            score = 4.0 * metrics["false_safe_rate"] + (1.0 - metrics["risk_accuracy"]) + 0.5 * metrics["error_mae"]
            if score < best_score:
                best_score = score
                best_error = error_guard
                best_prob = prob_guard
    return best_error, best_prob


def fit_model(
    *,
    train_samples: list[dict],
    val_samples: list[dict],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[TrustMLP, torch.Tensor, torch.Tensor, float, float]:
    train_features, _, _ = collate(train_samples)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    def normalize(samples: list[dict]) -> list[dict]:
        return [{**sample, "features": (sample["features"] - feature_mean) / feature_std} for sample in samples]

    train_norm = normalize(train_samples)
    val_norm = normalize(val_samples)

    train_loader = make_loader(train_norm, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_norm, batch_size=batch_size, shuffle=False)

    class_counts = Counter(sample["risk_label"] for sample in train_norm)
    class_weights = torch.tensor(
        [1.0 / class_counts.get(label, 1) for label in TRUST_LABELS],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.sum() * len(TRUST_LABELS)

    model = TrustMLP(input_dim=20, hidden_dim=64, num_classes=len(TRUST_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    risk_loss = nn.CrossEntropyLoss(weight=class_weights)
    error_loss = nn.L1Loss()

    best_state = None
    best_val = float("inf")
    best_error_guard = 0.1
    best_prob_guard = 0.5
    patience = 30
    stale = 0
    for _ in range(epochs):
        model.train()
        for features, risk_y, error_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = risk_loss(outputs["risk_logits"], risk_y) + 0.5 * error_loss(outputs["error_pred"], error_y)
            loss.backward()
            optimizer.step()

        error_guard, prob_guard = select_safe_guards(model, val_loader)
        metrics = evaluate(model, val_loader, safe_error_guard=error_guard, safe_prob_guard=prob_guard)
        val_score = (1.0 - metrics["risk_accuracy"]) + metrics["error_mae"] + 2.0 * metrics["false_safe_rate"]
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_error_guard = error_guard
            best_prob_guard = prob_guard
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, feature_mean, feature_std, best_error_guard, best_prob_guard


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
    train_samples, test_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=19,
        stratify=stratify_keys(samples),
    )
    train_samples, val_samples = train_test_split(
        train_samples,
        test_size=0.25,
        random_state=29,
        stratify=stratify_keys(train_samples),
    )

    def normalize(samples: list[dict]) -> list[dict]:
        return [{**sample, "features": (sample["features"] - feature_mean) / feature_std} for sample in samples]
    model, feature_mean, feature_std, safe_error_guard, safe_prob_guard = fit_model(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    train_samples = normalize(train_samples)
    val_samples = normalize(val_samples)
    test_samples = normalize(test_samples)

    train_loader = make_loader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_samples, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_samples, batch_size=batch_size, shuffle=False)

    metrics = {
        "train": evaluate(model, train_loader, safe_error_guard=safe_error_guard, safe_prob_guard=safe_prob_guard),
        "val": evaluate(model, val_loader, safe_error_guard=safe_error_guard, safe_prob_guard=safe_prob_guard),
        "test": evaluate(model, test_loader, safe_error_guard=safe_error_guard, safe_prob_guard=safe_prob_guard),
        "labels": TRUST_LABELS,
        "safe_error_guard": safe_error_guard,
        "safe_prob_guard": safe_prob_guard,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": {"input_dim": 20, "hidden_dim": 64, "num_classes": len(TRUST_LABELS)},
            "labels": TRUST_LABELS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "safe_error_guard": safe_error_guard,
            "safe_prob_guard": safe_prob_guard,
        },
        model_out,
    )
    two_by_two = [sample for sample in samples if sample["metadata"]["nsites"] == 4]
    two_by_three = [sample for sample in samples if sample["metadata"]["nsites"] == 6]
    if two_by_two and two_by_three:
        train_2x2_raw, val_2x2_raw = train_test_split(
            two_by_two,
            test_size=0.2,
            random_state=41,
            stratify=stratify_keys(two_by_two),
        )
        cross_model, cross_mean, cross_std, cross_error_guard, cross_prob_guard = fit_model(
            train_samples=train_2x2_raw,
            val_samples=val_2x2_raw,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        test_2x3 = [{**sample, "features": (sample["features"] - cross_mean) / cross_std} for sample in two_by_three]
        cross_loader = make_loader(test_2x3, batch_size=batch_size, shuffle=False)
        cross_metrics = evaluate(
            cross_model,
            cross_loader,
            safe_error_guard=cross_error_guard,
            safe_prob_guard=cross_prob_guard,
        )
        metrics["cross_lattice"] = {
            "train_lattice": "2x2",
            "test_lattice": "2x3",
            **cross_metrics,
            "num_train_samples": len(train_2x2_raw),
            "num_test_samples": len(test_2x3),
        }
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TRUST_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_TRUST_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_TRUST_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=300)
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

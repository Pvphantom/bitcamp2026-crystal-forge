from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.routing_model import RoutingMLP
from app.ml.schema import (
    ARTIFACTS_DIR,
    DEFAULT_ROUTING_DATASET,
    DEFAULT_ROUTING_METRICS_PATH,
    DEFAULT_ROUTING_MODEL_PATH,
    ROUTING_LABELS,
    ROUTING_TO_INDEX,
)


def collate(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    route_labels = torch.tensor([ROUTING_TO_INDEX[sample["route_label"]] for sample in samples], dtype=torch.long)
    return features, route_labels


def make_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features, route_labels = collate(samples)
    dataset = TensorDataset(features, route_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def filter_samples(
    samples: list[dict],
    *,
    allowed_reference_qualities: set[str],
    include_uncertain: bool,
) -> list[dict]:
    filtered = []
    for sample in samples:
        if sample.get("reference_quality") not in allowed_reference_qualities:
            continue
        if not include_uncertain and sample["route_label"] == "uncertain":
            continue
        filtered.append(sample)
    return filtered


def stratify_keys(samples: list[dict]) -> list[str] | None:
    counts = Counter(sample["route_label"] for sample in samples)
    if not counts or min(counts.values()) < 2:
        return None
    return [sample["route_label"] for sample in samples]


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    stratify = stratify_keys(samples)
    train_samples, test_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=31,
        stratify=stratify,
    )
    stratify_train = stratify_keys(train_samples)
    train_samples, val_samples = train_test_split(
        train_samples,
        test_size=0.25,
        random_state=37,
        stratify=stratify_train,
    )
    return train_samples, val_samples, test_samples


def evaluate(
    model: RoutingMLP,
    loader: DataLoader,
    *,
    confidence_threshold: float,
    ood_distance_threshold: float,
) -> dict[str, object]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    abstained = 0
    ood_abstained = 0
    low_conf_abstained = 0
    mean_field_predictions = 0
    false_mean_field_predictions = 0
    with torch.no_grad():
        for features, route_y in loader:
            outputs = model(features)
            probs = torch.softmax(outputs["route_logits"], dim=-1)
            confidences, pred_indices = torch.max(probs, dim=-1)
            distances = torch.linalg.norm(features, dim=-1)
            for truth, pred_idx, confidence, distance in zip(
                route_y.tolist(),
                pred_indices.tolist(),
                confidences.tolist(),
                distances.tolist(),
                strict=True,
            ):
                abstain_reason = None
                if float(distance) > ood_distance_threshold:
                    abstain_reason = "ood"
                elif float(confidence) < confidence_threshold:
                    abstain_reason = "low_confidence"
                if abstain_reason is not None:
                    predicted = ROUTING_TO_INDEX["uncertain"]
                    abstained += 1
                    if abstain_reason == "ood":
                        ood_abstained += 1
                    else:
                        low_conf_abstained += 1
                else:
                    predicted = pred_idx
                    if ROUTING_LABELS[predicted] == "mean_field":
                        mean_field_predictions += 1
                        if truth != ROUTING_TO_INDEX["mean_field"]:
                            false_mean_field_predictions += 1
                true_labels.append(truth)
                predicted_labels.append(predicted)

    total = len(true_labels)
    correct = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=True) if truth == pred)
    covered = total - abstained
    covered_correct = sum(
        1
        for truth, pred in zip(true_labels, predicted_labels, strict=True)
        if pred != ROUTING_TO_INDEX["uncertain"] and truth == pred
    )
    false_mean_field_rate = (
        0.0 if mean_field_predictions == 0 else false_mean_field_predictions / mean_field_predictions
    )
    return {
        "route_accuracy": 0.0 if total == 0 else correct / total,
        "covered_accuracy": 0.0 if covered == 0 else covered_correct / covered,
        "coverage": 0.0 if total == 0 else covered / total,
        "abstention_rate": 0.0 if total == 0 else abstained / total,
        "ood_abstention_rate": 0.0 if total == 0 else ood_abstained / total,
        "low_confidence_abstention_rate": 0.0 if total == 0 else low_conf_abstained / total,
        "false_mean_field_rate": false_mean_field_rate,
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=list(range(len(ROUTING_LABELS))),
        ).tolist(),
    }


def select_confidence_threshold(
    model: RoutingMLP,
    loader: DataLoader,
    *,
    ood_distance_threshold: float,
) -> float:
    candidates = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    best_threshold = candidates[0]
    best_score = float("inf")
    for threshold in candidates:
        metrics = evaluate(
            model,
            loader,
            confidence_threshold=threshold,
            ood_distance_threshold=ood_distance_threshold,
        )
        score = (
            4.0 * metrics["false_mean_field_rate"]
            + (1.0 - metrics["route_accuracy"])
            + 0.25 * metrics["abstention_rate"]
        )
        if score < best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def fit_model(
    *,
    train_samples: list[dict],
    val_samples: list[dict],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[RoutingMLP, torch.Tensor, torch.Tensor, float, float]:
    train_features, train_labels = collate(train_samples)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)

    def normalize(samples: list[dict]) -> list[dict]:
        return [{**sample, "features": (sample["features"] - feature_mean) / feature_std} for sample in samples]

    train_norm = normalize(train_samples)
    val_norm = normalize(val_samples)
    train_loader = make_loader(train_norm, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_norm, batch_size=batch_size, shuffle=False)

    label_counts = Counter(sample["route_label"] for sample in train_norm)
    class_weights = torch.tensor(
        [1.0 / label_counts.get(label, 1) for label in ROUTING_LABELS],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.sum() * len(ROUTING_LABELS)

    model = RoutingMLP(input_dim=int(train_features.shape[-1]), hidden_dim=64, num_routes=len(ROUTING_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    route_loss = nn.CrossEntropyLoss(weight=class_weights)

    normalized_train_features, _ = collate(train_norm)
    train_distances = torch.linalg.norm(normalized_train_features, dim=-1)
    ood_distance_threshold = float(torch.max(train_distances).item() * 1.1 + 1e-6)

    best_state = None
    best_val = float("inf")
    best_confidence_threshold = 0.55
    patience = 30
    stale = 0

    for _ in range(epochs):
        model.train()
        for features, route_y in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = route_loss(outputs["route_logits"], route_y)
            loss.backward()
            optimizer.step()

        confidence_threshold = select_confidence_threshold(
            model,
            val_loader,
            ood_distance_threshold=ood_distance_threshold,
        )
        metrics = evaluate(
            model,
            val_loader,
            confidence_threshold=confidence_threshold,
            ood_distance_threshold=ood_distance_threshold,
        )
        val_score = (
            4.0 * metrics["false_mean_field_rate"]
            + (1.0 - metrics["route_accuracy"])
            + 0.25 * metrics["abstention_rate"]
        )
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_confidence_threshold = confidence_threshold
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, feature_mean, feature_std, best_confidence_threshold, ood_distance_threshold


def train(
    *,
    dataset_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    allow_weak_labels: bool,
    include_uncertain: bool,
) -> dict[str, object]:
    raw_samples = torch.load(dataset_path, map_location="cpu")
    allowed_reference_qualities = {"strong"} | ({"weak"} if allow_weak_labels else set())
    samples = filter_samples(
        raw_samples,
        allowed_reference_qualities=allowed_reference_qualities,
        include_uncertain=include_uncertain,
    )
    if len(samples) < 3:
        raise ValueError("Not enough routing samples after provenance filtering to train a model")

    train_samples, val_samples, test_samples = split_samples(samples)

    model, feature_mean, feature_std, confidence_threshold, ood_distance_threshold = fit_model(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    def normalize(samples_to_normalize: list[dict]) -> list[dict]:
        return [
            {**sample, "features": (sample["features"] - feature_mean) / feature_std}
            for sample in samples_to_normalize
        ]

    train_norm = normalize(train_samples)
    val_norm = normalize(val_samples)
    test_norm = normalize(test_samples)

    train_loader = make_loader(train_norm, batch_size=batch_size, shuffle=False)
    val_loader = make_loader(val_norm, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_norm, batch_size=batch_size, shuffle=False)

    metrics = {
        "train": evaluate(
            model,
            train_loader,
            confidence_threshold=confidence_threshold,
            ood_distance_threshold=ood_distance_threshold,
        ),
        "val": evaluate(
            model,
            val_loader,
            confidence_threshold=confidence_threshold,
            ood_distance_threshold=ood_distance_threshold,
        ),
        "test": evaluate(
            model,
            test_loader,
            confidence_threshold=confidence_threshold,
            ood_distance_threshold=ood_distance_threshold,
        ),
        "labels": ROUTING_LABELS,
        "confidence_threshold": confidence_threshold,
        "ood_distance_threshold": ood_distance_threshold,
        "input_dim": int(feature_mean.shape[0]),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "num_raw_samples": len(raw_samples),
        "num_filtered_samples": len(samples),
        "allowed_reference_qualities": sorted(allowed_reference_qualities),
        "include_uncertain": include_uncertain,
        "reference_quality_counts": dict(Counter(sample["reference_quality"] for sample in samples)),
        "route_label_counts": dict(Counter(sample["route_label"] for sample in samples)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": {
                "input_dim": int(feature_mean.shape[0]),
                "hidden_dim": 64,
                "num_routes": len(ROUTING_LABELS),
            },
            "labels": ROUTING_LABELS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "confidence_threshold": confidence_threshold,
            "ood_distance_threshold": ood_distance_threshold,
            "allowed_reference_qualities": sorted(allowed_reference_qualities),
            "include_uncertain": include_uncertain,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROUTING_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_ROUTING_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_ROUTING_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--include-uncertain", action="store_true")
    args = parser.parse_args()

    metrics = train(
        dataset_path=args.dataset,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        allow_weak_labels=args.allow_weak_labels,
        include_uncertain=args.include_uncertain,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

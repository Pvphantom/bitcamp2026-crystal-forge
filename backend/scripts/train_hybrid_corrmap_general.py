from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.analysis.intrinsic_feature_vector_general import (
    GENERAL_INTRINSIC_FEATURE_DIM,
    build_general_intrinsic_features,
)
from app.ml.hybrid_corrmap_model import HybridCorrMapMLP
from app.ml.schema import (
    ARTIFACTS_DIR,
    INTRINSIC_RISK_LABELS,
    INTRINSIC_RISK_TO_INDEX,
    ROUTING_LABELS,
    ROUTING_TO_INDEX,
)
from scripts.eval_corrmap_on_regime_benchmark_general import evaluate_regime_benchmark_general
from scripts.train_routing_model import filter_samples as filter_routing_samples


INTRINSIC_TO_ROUTE = {
    "stable_classical": "mean_field",
    "fragile_classical": "scalable_classical",
    "frontier_or_uncertain": "quantum_frontier",
}


def _split(samples: list[dict], label_key: str) -> tuple[list[dict], list[dict], list[dict]]:
    counts = Counter(sample[label_key] for sample in samples)
    stratify = None if not counts or min(counts.values()) < 2 else [sample[label_key] for sample in samples]
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=41, stratify=stratify)
    train_counts = Counter(sample[label_key] for sample in train_samples)
    stratify_train = None if not train_counts or min(train_counts.values()) < 2 else [sample[label_key] for sample in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=0.25, random_state=43, stratify=stratify_train)
    return train_samples, val_samples, test_samples


def _attach_general_features(sample: dict, *, allow_missing_intrinsic: bool) -> dict:
    if allow_missing_intrinsic and "stability" not in sample:
        base = torch.as_tensor(sample["features"], dtype=torch.float32)
        zeros = torch.zeros(GENERAL_INTRINSIC_FEATURE_DIM - int(base.shape[0]), dtype=torch.float32)
        return {**sample, "features": torch.cat([base, zeros], dim=0)}
    return {**sample, "features": build_general_intrinsic_features(sample)}


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _route_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    labels = torch.tensor([ROUTING_TO_INDEX[sample["route_label"]] for sample in samples], dtype=torch.long)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def _intrinsic_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    labels = torch.tensor([INTRINSIC_RISK_TO_INDEX[sample["intrinsic_label"]] for sample in samples], dtype=torch.long)
    pseudo_routes = torch.tensor(
        [
            ROUTING_TO_INDEX[
                sample.get("physical_tractability", {}).get("route_prior")
                if sample.get("physical_tractability", {}).get("route_prior") in ROUTING_TO_INDEX
                else INTRINSIC_TO_ROUTE[sample["intrinsic_label"]]
            ]
            for sample in samples
        ],
        dtype=torch.long,
    )
    return DataLoader(TensorDataset(features, labels, pseudo_routes), batch_size=batch_size, shuffle=shuffle)


def _classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((torch.argmax(logits, dim=-1) == labels).float().mean().item())


def train(
    *,
    routing_dataset_path: Path,
    intrinsic_dataset_path: Path,
    benchmark_dataset_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    raw_routing = torch.load(routing_dataset_path, map_location="cpu")
    routing_samples = filter_routing_samples(raw_routing, allowed_reference_qualities={"strong"}, include_uncertain=False)
    intrinsic_samples = torch.load(intrinsic_dataset_path, map_location="cpu")
    benchmark_samples = torch.load(benchmark_dataset_path, map_location="cpu")
    benchmark_6x6 = [sample for sample in benchmark_samples if int(sample["problem"]["Lx"]) == 6]

    routing_samples = [_attach_general_features(sample, allow_missing_intrinsic=True) for sample in routing_samples]
    intrinsic_samples = [_attach_general_features(sample, allow_missing_intrinsic=False) for sample in intrinsic_samples]

    route_train, route_val, route_test = _split(routing_samples, "route_label")
    intr_train, intr_val, intr_test = _split(intrinsic_samples, "intrinsic_label")

    normalization_features = torch.stack([sample["features"] for sample in route_train + intr_train], dim=0)
    feature_mean = normalization_features.mean(dim=0)
    feature_std = normalization_features.std(dim=0).clamp_min(1e-6)

    route_train = _normalize(route_train, feature_mean, feature_std)
    route_val = _normalize(route_val, feature_mean, feature_std)
    route_test = _normalize(route_test, feature_mean, feature_std)
    intr_train = _normalize(intr_train, feature_mean, feature_std)
    intr_val = _normalize(intr_val, feature_mean, feature_std)
    intr_test = _normalize(intr_test, feature_mean, feature_std)

    route_train_loader = _route_loader(route_train, batch_size=batch_size, shuffle=True)
    route_val_loader = _route_loader(route_val, batch_size=batch_size, shuffle=False)
    route_test_loader = _route_loader(route_test, batch_size=batch_size, shuffle=False)
    intr_train_loader = _intrinsic_loader(intr_train, batch_size=batch_size, shuffle=True)
    intr_val_loader = _intrinsic_loader(intr_val, batch_size=batch_size, shuffle=False)
    intr_test_loader = _intrinsic_loader(intr_test, batch_size=batch_size, shuffle=False)

    model = HybridCorrMapMLP(
        input_dim=int(feature_mean.shape[0]),
        hidden_dim=96,
        num_routes=len(ROUTING_LABELS),
        num_intrinsic_labels=len(INTRINSIC_RISK_LABELS),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    route_counts = Counter(sample["route_label"] for sample in route_train)
    route_weights = torch.tensor([1.0 / route_counts.get(label, 1) for label in ROUTING_LABELS], dtype=torch.float32)
    route_weights = route_weights / route_weights.sum() * len(ROUTING_LABELS)
    intrinsic_counts = Counter(sample["intrinsic_label"] for sample in intr_train)
    intrinsic_weights = torch.tensor([1.0 / intrinsic_counts.get(label, 1) for label in INTRINSIC_RISK_LABELS], dtype=torch.float32)
    intrinsic_weights = intrinsic_weights / intrinsic_weights.sum() * len(INTRINSIC_RISK_LABELS)
    route_loss = nn.CrossEntropyLoss(weight=route_weights)
    intrinsic_loss = nn.CrossEntropyLoss(weight=intrinsic_weights)

    normalized_train_features = torch.stack([sample["features"] for sample in route_train], dim=0)
    train_distances = torch.linalg.norm(normalized_train_features, dim=-1)
    ood_distance_threshold = float(torch.max(train_distances).item() * 1.1 + 1e-6)

    best_state = None
    best_score = float("-inf")
    best_confidence_threshold = 0.55
    patience = 30
    stale = 0

    for _ in range(epochs):
        model.train()
        intr_iter = iter(intr_train_loader)
        for route_features, route_labels in route_train_loader:
            try:
                intr_features, intr_labels, pseudo_route_labels = next(intr_iter)
            except StopIteration:
                intr_iter = iter(intr_train_loader)
                intr_features, intr_labels, pseudo_route_labels = next(intr_iter)
            optimizer.zero_grad()
            route_outputs = model(route_features)
            intr_outputs = model(intr_features)
            frontier_mask = torch.tensor(
                [1.0 if INTRINSIC_RISK_LABELS[index] == "frontier_or_uncertain" else 0.0 for index in intr_labels.tolist()],
                dtype=torch.float32,
            )
            route_probs_intr = torch.softmax(intr_outputs["route_logits"], dim=-1)
            mean_field_probs_intr = route_probs_intr[:, ROUTING_TO_INDEX["mean_field"]]
            scalable_probs_intr = route_probs_intr[:, ROUTING_TO_INDEX["scalable_classical"]]
            frontier_penalty = torch.mean(frontier_mask * (mean_field_probs_intr + 0.5 * scalable_probs_intr))
            loss = (
                route_loss(route_outputs["route_logits"], route_labels)
                + intrinsic_loss(intr_outputs["intrinsic_logits"], intr_labels)
                + 0.9 * route_loss(intr_outputs["route_logits"], pseudo_route_labels)
                + 0.8 * frontier_penalty
            )
            loss.backward()
            optimizer.step()

        confidence_threshold = _select_confidence_threshold(model, route_val_loader, ood_distance_threshold)
        temp_model_path = model_out.parent / f".tmp_{model_out.name}"
        torch.save(
            {
                "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                "model_config": {
                    "input_dim": int(feature_mean.shape[0]),
                    "hidden_dim": 96,
                    "num_routes": len(ROUTING_LABELS),
                    "num_intrinsic_labels": len(INTRINSIC_RISK_LABELS),
                },
                "route_labels": ROUTING_LABELS,
                "intrinsic_labels": INTRINSIC_RISK_LABELS,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "confidence_threshold": confidence_threshold,
                "ood_distance_threshold": ood_distance_threshold,
            },
            temp_model_path,
        )
        benchmark_report = evaluate_regime_benchmark_general(samples=benchmark_6x6, model_path=temp_model_path)
        benchmark_acc = float(benchmark_report["summary"]["overall_accuracy"])
        mean_field_f1 = float(benchmark_report["summary"]["per_class_metrics"].get("mean_field", {}).get("f1", 0.0))
        scalable_f1 = float(benchmark_report["summary"]["per_class_metrics"].get("scalable_classical", {}).get("f1", 0.0))
        quantum_f1 = float(benchmark_report["summary"]["per_class_metrics"].get("quantum_frontier", {}).get("f1", 0.0))
        score = benchmark_acc + 0.40 * mean_field_f1 + 0.25 * scalable_f1 + 0.15 * quantum_f1
        temp_model_path.unlink(missing_ok=True)

        if score > best_score:
            best_score = score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_confidence_threshold = confidence_threshold
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    metrics = {
        "routing_train_accuracy": _eval_route_accuracy(model, route_train_loader),
        "routing_val_accuracy": _eval_route_accuracy(model, route_val_loader),
        "routing_test_accuracy": _eval_route_accuracy(model, route_test_loader),
        "intrinsic_train_accuracy": _eval_intrinsic_accuracy(model, intr_train_loader),
        "intrinsic_val_accuracy": _eval_intrinsic_accuracy(model, intr_val_loader),
        "intrinsic_test_accuracy": _eval_intrinsic_accuracy(model, intr_test_loader),
        "benchmark_6x6_score": best_score,
        "confidence_threshold": best_confidence_threshold,
        "ood_distance_threshold": ood_distance_threshold,
        "num_routing_samples": len(routing_samples),
        "num_intrinsic_samples": len(intrinsic_samples),
        "training_lattices": {
            "routing": sorted({f"{sample['problem_metadata']['Lx']}x{sample['problem_metadata']['Ly']}" for sample in routing_samples}),
            "intrinsic": sorted({f"{sample['problem_metadata']['Lx']}x{sample['problem_metadata']['Ly']}" for sample in intrinsic_samples}),
            "benchmark_model_selection": ["6x6"],
            "benchmark_held_out": ["8x8"],
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": {
                "input_dim": int(feature_mean.shape[0]),
                "hidden_dim": 96,
                "num_routes": len(ROUTING_LABELS),
                "num_intrinsic_labels": len(INTRINSIC_RISK_LABELS),
            },
            "route_labels": ROUTING_LABELS,
            "intrinsic_labels": INTRINSIC_RISK_LABELS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "confidence_threshold": best_confidence_threshold,
            "ood_distance_threshold": ood_distance_threshold,
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def _eval_route_accuracy(model: HybridCorrMapMLP, loader: DataLoader) -> float:
    model.eval()
    accuracies = []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            accuracies.append(_classification_accuracy(outputs["route_logits"], labels))
    return float(sum(accuracies) / len(accuracies)) if accuracies else 0.0


def _eval_intrinsic_accuracy(model: HybridCorrMapMLP, loader: DataLoader) -> float:
    model.eval()
    accuracies = []
    with torch.no_grad():
        for batch in loader:
            features, labels = batch[0], batch[1]
            outputs = model(features)
            accuracies.append(_classification_accuracy(outputs["intrinsic_logits"], labels))
    return float(sum(accuracies) / len(accuracies)) if accuracies else 0.0


def _select_confidence_threshold(model: HybridCorrMapMLP, loader: DataLoader, ood_distance_threshold: float) -> float:
    candidates = [0.35, 0.45, 0.55, 0.65, 0.75]
    best = candidates[0]
    best_score = float("inf")
    model.eval()
    for threshold in candidates:
        false_mean_field = 0
        mean_field_predictions = 0
        total = 0
        with torch.no_grad():
            for features, labels in loader:
                outputs = model(features)
                probs = torch.softmax(outputs["route_logits"], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1).values
                distances = torch.linalg.norm(features, dim=-1)
                for p, truth, conf, dist in zip(preds.tolist(), labels.tolist(), confidences.tolist(), distances.tolist(), strict=True):
                    total += 1
                    if dist > ood_distance_threshold or conf < threshold:
                        continue
                    if ROUTING_LABELS[p] == "mean_field":
                        mean_field_predictions += 1
                        if ROUTING_LABELS[truth] != "mean_field":
                            false_mean_field += 1
        false_rate = 0.0 if mean_field_predictions == 0 else false_mean_field / mean_field_predictions
        score = false_rate + 0.05 * threshold
        if score < best_score:
            best = threshold
            best_score = score
    return float(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dataset", type=Path, default=Path("backend/artifacts/routing_dataset_test.pt"))
    parser.add_argument("--intrinsic-dataset", type=Path, default=Path("backend/artifacts/intrinsic_corrmap_dataset_general.pt"))
    parser.add_argument("--benchmark-dataset", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/hybrid_corrmap_general.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/hybrid_corrmap_general.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train(
        routing_dataset_path=args.routing_dataset,
        intrinsic_dataset_path=args.intrinsic_dataset,
        benchmark_dataset_path=args.benchmark_dataset,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

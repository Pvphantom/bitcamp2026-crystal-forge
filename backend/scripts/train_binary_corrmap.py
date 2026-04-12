from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.analysis.intrinsic_feature_vector import build_intrinsic_augmented_features
from app.ml.binary_corrmap_model import BinaryCorrMapMLP
from app.ml.schema import ARTIFACTS_DIR
from scripts.eval_binary_corrmap_on_regime_benchmark import evaluate_binary_regime_benchmark
from scripts.train_routing_model import filter_samples as filter_routing_samples


POSITIVE_LABEL = "quantum_frontier"
NEGATIVE_LABEL = "classical_scalable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dataset", type=Path, default=Path("backend/artifacts/routing_dataset_test.pt"))
    parser.add_argument("--intrinsic-dataset", type=Path, default=Path("backend/artifacts/intrinsic_corrmap_dataset_test.pt"))
    parser.add_argument("--benchmark-dataset", type=Path, default=Path("backend/artifacts/regime_benchmark_refined_v2.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/binary_corrmap.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/binary_corrmap.json"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--selection-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
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
        selection_every=args.selection_every,
        patience=args.patience,
    )
    print(json.dumps(metrics, indent=2))


def _binary_label_from_route(label: str) -> str:
    return POSITIVE_LABEL if label == POSITIVE_LABEL else NEGATIVE_LABEL


def _binary_label_from_intrinsic(label: str) -> str:
    return POSITIVE_LABEL if label == "frontier_or_uncertain" else NEGATIVE_LABEL


def _attach_augmented_features(sample: dict, *, allow_missing_intrinsic: bool) -> dict:
    if allow_missing_intrinsic and "stability" not in sample:
        base = torch.as_tensor(sample["features"], dtype=torch.float32)
        zeros = torch.zeros(27, dtype=torch.float32)
        return {**sample, "features": torch.cat([base, zeros], dim=0)}
    return {**sample, "features": build_intrinsic_augmented_features(sample)}


def _split(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [sample["binary_label"] for sample in samples]
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=41, stratify=labels)
    train_labels = [sample["binary_label"] for sample in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=0.25, random_state=43, stratify=train_labels)
    return train_samples, val_samples, test_samples


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    labels = torch.tensor([1.0 if sample["binary_label"] == POSITIVE_LABEL else 0.0 for sample in samples], dtype=torch.float32)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


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
    selection_every: int,
    patience: int,
) -> dict[str, object]:
    raw_routing = torch.load(routing_dataset_path, map_location="cpu")
    routing_samples = filter_routing_samples(raw_routing, allowed_reference_qualities={"strong"}, include_uncertain=False)
    routing_samples = [
        {**sample, "binary_label": _binary_label_from_route(sample["route_label"])}
        for sample in routing_samples
    ]
    intrinsic_samples = torch.load(intrinsic_dataset_path, map_location="cpu")
    intrinsic_samples = [
        {**sample, "binary_label": _binary_label_from_intrinsic(sample["intrinsic_label"])}
        for sample in intrinsic_samples
    ]
    benchmark_samples = torch.load(benchmark_dataset_path, map_location="cpu")
    benchmark_6x6 = [sample for sample in benchmark_samples if int(sample["problem"]["Lx"]) == 6]

    routing_samples = [_attach_augmented_features(sample, allow_missing_intrinsic=True) for sample in routing_samples]
    intrinsic_samples = [_attach_augmented_features(sample, allow_missing_intrinsic=False) for sample in intrinsic_samples]

    route_train, route_val, route_test = _split(routing_samples)
    intr_train, intr_val, intr_test = _split(intrinsic_samples)

    norm_features = torch.stack([sample["features"] for sample in route_train + intr_train], dim=0)
    feature_mean = norm_features.mean(dim=0)
    feature_std = norm_features.std(dim=0).clamp_min(1e-6)

    route_train = _normalize(route_train, feature_mean, feature_std)
    route_val = _normalize(route_val, feature_mean, feature_std)
    route_test = _normalize(route_test, feature_mean, feature_std)
    intr_train = _normalize(intr_train, feature_mean, feature_std)
    intr_val = _normalize(intr_val, feature_mean, feature_std)
    intr_test = _normalize(intr_test, feature_mean, feature_std)

    train_loader = _loader(route_train + intr_train, batch_size=batch_size, shuffle=True)
    route_val_loader = _loader(route_val, batch_size=batch_size, shuffle=False)
    route_test_loader = _loader(route_test, batch_size=batch_size, shuffle=False)
    intr_val_loader = _loader(intr_val, batch_size=batch_size, shuffle=False)
    intr_test_loader = _loader(intr_test, batch_size=batch_size, shuffle=False)

    model = BinaryCorrMapMLP(input_dim=int(feature_mean.shape[0]), hidden_dim=96)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    class_counts = Counter(sample["binary_label"] for sample in route_train + intr_train)
    pos_weight = torch.tensor([class_counts[NEGATIVE_LABEL] / max(class_counts[POSITIVE_LABEL], 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_score = float("-inf")
    best_threshold = 0.5
    stale = 0
    patience = max(1, patience)

    for _ in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        if (_ + 1) % max(1, selection_every) != 0 and (_ + 1) != epochs:
            continue
        threshold = _select_threshold(model, route_val_loader, intr_val_loader)
        tmp_model = model_out.parent / f".tmp_{model_out.name}"
        torch.save(
            {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": {"input_dim": int(feature_mean.shape[0]), "hidden_dim": 96},
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "threshold": threshold,
                "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
                "model_type": "binary_corrmap",
            },
            tmp_model,
        )
        benchmark_report = evaluate_binary_regime_benchmark(samples=benchmark_6x6, model_path=tmp_model)
        score = float(benchmark_report["summary"]["overall_accuracy"])
        tmp_model.unlink(missing_ok=True)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    metrics = {
        "routing_val_accuracy": _eval_accuracy(model, route_val_loader, best_threshold),
        "routing_test_accuracy": _eval_accuracy(model, route_test_loader, best_threshold),
        "intrinsic_val_accuracy": _eval_accuracy(model, intr_val_loader, best_threshold),
        "intrinsic_test_accuracy": _eval_accuracy(model, intr_test_loader, best_threshold),
        "benchmark_6x6_accuracy": best_score,
        "threshold": best_threshold,
        "num_routing_samples": len(routing_samples),
        "num_intrinsic_samples": len(intrinsic_samples),
        "training_lattices": {
            "routing": sorted({f"{s['problem_metadata']['Lx']}x{s['problem_metadata']['Ly']}" for s in routing_samples}),
            "intrinsic": sorted({f"{s['problem_metadata']['Lx']}x{s['problem_metadata']['Ly']}" for s in intrinsic_samples}),
            "benchmark_model_selection": ["6x6"],
            "benchmark_held_out": ["8x8"],
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {"input_dim": int(feature_mean.shape[0]), "hidden_dim": 96},
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": best_threshold,
            "labels": [NEGATIVE_LABEL, POSITIVE_LABEL],
            "model_type": "binary_corrmap",
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def _eval_accuracy(model: BinaryCorrMapMLP, loader: DataLoader, threshold: float) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in loader:
            preds = (torch.sigmoid(model(features)) >= threshold).float()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return 0.0 if total == 0 else correct / total


def _select_threshold(model: BinaryCorrMapMLP, route_loader: DataLoader, intr_loader: DataLoader) -> float:
    candidates = [0.35, 0.45, 0.5, 0.55, 0.65]
    best_t = 0.5
    best_score = float("-inf")
    for threshold in candidates:
        score = _eval_accuracy(model, route_loader, threshold) + _eval_accuracy(model, intr_loader, threshold)
        if score > best_score:
            best_score = score
            best_t = threshold
    return best_t


if __name__ == "__main__":
    main()

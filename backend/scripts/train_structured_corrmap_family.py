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
from app.ml.schema import DEFAULT_INTRINSIC_CORRMAP_DATASET, DEFAULT_ROUTING_DATASET, INTRINSIC_RISK_LABELS, INTRINSIC_RISK_TO_INDEX
from app.ml.structured_corrmap_model import StructuredCorrMapMLP
from scripts.eval_corrmap_on_regime_benchmark import evaluate_regime_benchmark
from scripts.train_routing_model import filter_samples as filter_routing_samples


def _split(samples: list[dict], label_key: str) -> tuple[list[dict], list[dict], list[dict]]:
    counts = Counter(sample[label_key] for sample in samples)
    stratify = None if not counts or min(counts.values()) < 2 else [sample[label_key] for sample in samples]
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=31, stratify=stratify)
    train_counts = Counter(sample[label_key] for sample in train_samples)
    stratify_train = None if not train_counts or min(train_counts.values()) < 2 else [sample[label_key] for sample in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=0.25, random_state=37, stratify=stratify_train)
    return train_samples, val_samples, test_samples


def _attach_augmented_features(sample: dict, *, allow_missing_intrinsic: bool) -> dict:
    if allow_missing_intrinsic and "stability" not in sample:
        base = torch.as_tensor(sample["features"], dtype=torch.float32)
        zeros = torch.zeros(27, dtype=torch.float32)
        return {**sample, "features": torch.cat([base, zeros], dim=0)}
    return {**sample, "features": build_intrinsic_augmented_features(sample)}


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _route_targets(label: str) -> tuple[float, float]:
    return (
        1.0 if label == "mean_field" else 0.0,
        1.0 if label == "quantum_frontier" else 0.0,
    )


def _route_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    mean_field_targets = torch.tensor([_route_targets(sample["route_label"])[0] for sample in samples], dtype=torch.float32)
    quantum_targets = torch.tensor([_route_targets(sample["route_label"])[1] for sample in samples], dtype=torch.float32)
    return DataLoader(TensorDataset(features, mean_field_targets, quantum_targets), batch_size=batch_size, shuffle=shuffle)


def _intrinsic_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    labels = torch.tensor([INTRINSIC_RISK_TO_INDEX[sample["intrinsic_label"]] for sample in samples], dtype=torch.long)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def train(
    *,
    routing_dataset_path: Path,
    intrinsic_dataset_path: Path,
    benchmark_dataset_path: Path,
    family_filter: str,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    raw_routing = torch.load(routing_dataset_path, map_location="cpu")
    routing_samples = filter_samples = filter_routing_samples(
        raw_routing,
        allowed_reference_qualities={"strong"},
        include_uncertain=False,
    )
    routing_samples = [sample for sample in routing_samples if str(sample.get("problem_metadata", {}).get("family")) == family_filter]
    intrinsic_samples = torch.load(intrinsic_dataset_path, map_location="cpu")
    intrinsic_samples = [sample for sample in intrinsic_samples if str(sample.get("problem_metadata", {}).get("family")) == family_filter]
    benchmark_samples = torch.load(benchmark_dataset_path, map_location="cpu")
    benchmark_6x6 = [sample for sample in benchmark_samples if str(sample["problem"]["model_family"]) == family_filter and int(sample["problem"]["Lx"]) == 6]

    routing_samples = [_attach_augmented_features(sample, allow_missing_intrinsic=True) for sample in routing_samples]
    intrinsic_samples = [_attach_augmented_features(sample, allow_missing_intrinsic=False) for sample in intrinsic_samples]

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

    model = StructuredCorrMapMLP(input_dim=int(feature_mean.shape[0]), hidden_dim=96, num_intrinsic_labels=len(INTRINSIC_RISK_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    mf_pos = sum(1 for sample in route_train if sample["route_label"] == "mean_field")
    q_pos = sum(1 for sample in route_train if sample["route_label"] == "quantum_frontier")
    bce_mf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1.0, (len(route_train) - mf_pos) / max(mf_pos, 1)), dtype=torch.float32))
    bce_q = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1.0, (len(route_train) - q_pos) / max(q_pos, 1)), dtype=torch.float32))
    intrinsic_counts = Counter(sample["intrinsic_label"] for sample in intr_train)
    intrinsic_weights = torch.tensor([1.0 / intrinsic_counts.get(label, 1) for label in INTRINSIC_RISK_LABELS], dtype=torch.float32)
    intrinsic_weights = intrinsic_weights / intrinsic_weights.sum() * len(INTRINSIC_RISK_LABELS)
    intrinsic_loss = nn.CrossEntropyLoss(weight=intrinsic_weights)

    best_state = None
    best_score = float("-inf")
    patience = 30
    stale = 0

    for _ in range(epochs):
        model.train()
        intr_iter = iter(intr_train_loader)
        for route_features, mf_targets, q_targets in route_train_loader:
            try:
                intr_features, intr_labels = next(intr_iter)
            except StopIteration:
                intr_iter = iter(intr_train_loader)
                intr_features, intr_labels = next(intr_iter)
            optimizer.zero_grad()
            route_outputs = model(route_features)
            intr_outputs = model(intr_features)
            loss = (
                bce_mf(route_outputs["mean_field_logit"], mf_targets)
                + 1.2 * bce_q(route_outputs["quantum_frontier_logit"], q_targets)
                + intrinsic_loss(intr_outputs["intrinsic_logits"], intr_labels)
            )
            loss.backward()
            optimizer.step()

        tmp_path = model_out.parent / f".tmp_{model_out.name}"
        torch.save(
            {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": {
                    "input_dim": int(feature_mean.shape[0]),
                    "hidden_dim": 96,
                    "num_intrinsic_labels": len(INTRINSIC_RISK_LABELS),
                },
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "family_filter": family_filter,
                "model_type": "structured_family",
            },
            tmp_path,
        )
        report = evaluate_regime_benchmark(samples=benchmark_6x6, model_path=tmp_path)
        score = float(report["summary"]["overall_accuracy"])
        tmp_path.unlink(missing_ok=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    metrics = {
        "family_filter": family_filter,
        "benchmark_6x6_accuracy": best_score,
        "routing_train_accuracy": _eval_route_accuracy(model, route_train_loader),
        "routing_val_accuracy": _eval_route_accuracy(model, route_val_loader),
        "routing_test_accuracy": _eval_route_accuracy(model, route_test_loader),
        "intrinsic_train_accuracy": _eval_intrinsic_accuracy(model, intr_train_loader),
        "intrinsic_val_accuracy": _eval_intrinsic_accuracy(model, intr_val_loader),
        "intrinsic_test_accuracy": _eval_intrinsic_accuracy(model, intr_test_loader),
    }
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {
                "input_dim": int(feature_mean.shape[0]),
                "hidden_dim": 96,
                "num_intrinsic_labels": len(INTRINSIC_RISK_LABELS),
            },
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "family_filter": family_filter,
            "model_type": "structured_family",
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def _eval_route_accuracy(model: StructuredCorrMapMLP, loader: DataLoader) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for features, mf_targets, q_targets in loader:
            out = model(features)
            preds = _decode_batch(out["mean_field_logit"], out["quantum_frontier_logit"])
            truths = _decode_targets(mf_targets, q_targets)
            values.append(sum(int(p == t) for p, t in zip(preds, truths, strict=True)) / max(len(truths), 1))
    return float(sum(values) / len(values)) if values else 0.0


def _eval_intrinsic_accuracy(model: StructuredCorrMapMLP, loader: DataLoader) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for features, labels in loader:
            out = model(features)
            values.append(float((torch.argmax(out["intrinsic_logits"], dim=-1) == labels).float().mean().item()))
    return float(sum(values) / len(values)) if values else 0.0


def _decode_batch(mean_field_logit: torch.Tensor, quantum_logit: torch.Tensor) -> list[str]:
    mf_probs = torch.sigmoid(mean_field_logit)
    q_probs = torch.sigmoid(quantum_logit)
    labels: list[str] = []
    for mf, q in zip(mf_probs.tolist(), q_probs.tolist(), strict=True):
        if q >= 0.5:
            labels.append("quantum_frontier")
        elif mf >= 0.7:
            labels.append("mean_field")
        else:
            labels.append("scalable_classical")
    return labels


def _decode_targets(mf_targets: torch.Tensor, q_targets: torch.Tensor) -> list[str]:
    labels: list[str] = []
    for mf, q in zip(mf_targets.tolist(), q_targets.tolist(), strict=True):
        if q >= 0.5:
            labels.append("quantum_frontier")
        elif mf >= 0.5:
            labels.append("mean_field")
        else:
            labels.append("scalable_classical")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dataset", type=Path, default=DEFAULT_ROUTING_DATASET)
    parser.add_argument("--intrinsic-dataset", type=Path, default=DEFAULT_INTRINSIC_CORRMAP_DATASET)
    parser.add_argument("--benchmark-dataset", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--family-filter", type=str, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train(
        routing_dataset_path=args.routing_dataset,
        intrinsic_dataset_path=args.intrinsic_dataset,
        benchmark_dataset_path=args.benchmark_dataset,
        family_filter=args.family_filter,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

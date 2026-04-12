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
from app.ml.schema import INTRINSIC_RISK_LABELS, INTRINSIC_RISK_TO_INDEX, ROUTING_LABELS
from app.ml.structured_corrmap_model import StructuredCorrMapMLP
from scripts.eval_corrmap_on_regime_benchmark_structured import evaluate_regime_benchmark_structured
from scripts.train_routing_model import filter_samples as filter_routing_samples


INTRINSIC_TO_BINARY = {
    "stable_classical": (1.0, 0.0),
    "fragile_classical": (0.0, 0.0),
    "frontier_or_uncertain": (0.0, 1.0),
}


def _split(samples: list[dict], label_key: str) -> tuple[list[dict], list[dict], list[dict]]:
    counts = Counter(sample[label_key] for sample in samples)
    stratify = None if not counts or min(counts.values()) < 2 else [sample[label_key] for sample in samples]
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=41, stratify=stratify)
    train_counts = Counter(sample[label_key] for sample in train_samples)
    stratify_train = None if not train_counts or min(train_counts.values()) < 2 else [sample[label_key] for sample in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=0.25, random_state=43, stratify=stratify_train)
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
    labels = torch.tensor([ROUTING_LABELS.index(sample["route_label"]) for sample in samples], dtype=torch.long)
    return DataLoader(TensorDataset(features, mean_field_targets, quantum_targets, labels), batch_size=batch_size, shuffle=shuffle)


def _intrinsic_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.stack([sample["features"] for sample in samples], dim=0)
    intrinsic_labels = torch.tensor([INTRINSIC_RISK_TO_INDEX[sample["intrinsic_label"]] for sample in samples], dtype=torch.long)
    mf_targets = torch.tensor([INTRINSIC_TO_BINARY[sample["intrinsic_label"]][0] for sample in samples], dtype=torch.float32)
    q_targets = torch.tensor([INTRINSIC_TO_BINARY[sample["intrinsic_label"]][1] for sample in samples], dtype=torch.float32)
    return DataLoader(TensorDataset(features, intrinsic_labels, mf_targets, q_targets), batch_size=batch_size, shuffle=shuffle)


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

    normalized_train_features = torch.stack([sample["features"] for sample in route_train], dim=0)
    ood_distance_threshold = float(torch.max(torch.linalg.norm(normalized_train_features, dim=-1)).item() * 1.1 + 1e-6)

    best_state = None
    best_score = float("-inf")
    best_thresholds = (0.72, 0.5)
    patience = 12
    stale = 0

    for _ in range(epochs):
        model.train()
        intr_iter = iter(intr_train_loader)
        for route_features, mf_targets, q_targets, route_labels in route_train_loader:
            try:
                intr_features, intr_labels, intr_mf_targets, intr_q_targets = next(intr_iter)
            except StopIteration:
                intr_iter = iter(intr_train_loader)
                intr_features, intr_labels, intr_mf_targets, intr_q_targets = next(intr_iter)
            optimizer.zero_grad()
            route_outputs = model(route_features)
            intr_outputs = model(intr_features)
            route_probs = torch.sigmoid(route_outputs["mean_field_logit"])
            quantum_probs = torch.sigmoid(route_outputs["quantum_frontier_logit"])
            invalid_overlap = torch.mean(route_probs * quantum_probs)
            false_mid_penalty = torch.mean(
                (route_labels == ROUTING_LABELS.index("scalable_classical")).float()
                * torch.abs(route_probs - 0.5)
                * torch.abs(quantum_probs - 0.5)
            )
            loss = (
                bce_mf(route_outputs["mean_field_logit"], mf_targets)
                + 1.2 * bce_q(route_outputs["quantum_frontier_logit"], q_targets)
                + intrinsic_loss(intr_outputs["intrinsic_logits"], intr_labels)
                + 0.5 * bce_mf(intr_outputs["mean_field_logit"], intr_mf_targets)
                + 0.8 * bce_q(intr_outputs["quantum_frontier_logit"], intr_q_targets)
                + 0.4 * invalid_overlap
                + 0.2 * false_mid_penalty
            )
            loss.backward()
            optimizer.step()

        score = _validation_score(model, route_val_loader, intr_val_loader)
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
    mean_field_threshold, quantum_threshold, benchmark_score = _select_thresholds(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        benchmark_6x6=benchmark_6x6,
        temp_model_path=model_out.parent / f".tmp_{model_out.name}",
        ood_distance_threshold=ood_distance_threshold,
    )

    metrics = {
        "routing_train_accuracy": _eval_route_accuracy(model, route_train_loader, mean_field_threshold, quantum_threshold),
        "routing_val_accuracy": _eval_route_accuracy(model, route_val_loader, mean_field_threshold, quantum_threshold),
        "routing_test_accuracy": _eval_route_accuracy(model, route_test_loader, mean_field_threshold, quantum_threshold),
        "intrinsic_train_accuracy": _eval_intrinsic_accuracy(model, intr_train_loader),
        "intrinsic_val_accuracy": _eval_intrinsic_accuracy(model, intr_val_loader),
        "intrinsic_test_accuracy": _eval_intrinsic_accuracy(model, intr_test_loader),
        "benchmark_6x6_score": benchmark_score,
        "mean_field_threshold": mean_field_threshold,
        "quantum_threshold": quantum_threshold,
        "ood_distance_threshold": ood_distance_threshold,
        "training_lattices": {
            "routing": sorted({f"{sample['problem_metadata']['Lx']}x{sample['problem_metadata']['Ly']}" for sample in routing_samples}),
            "intrinsic": sorted({f"{sample['problem_metadata']['Lx']}x{sample['problem_metadata']['Ly']}" for sample in intrinsic_samples}),
            "benchmark_model_selection": ["6x6"],
            "benchmark_held_out": ["8x8"],
        },
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
            "mean_field_threshold": mean_field_threshold,
            "quantum_threshold": quantum_threshold,
            "ood_distance_threshold": ood_distance_threshold,
            "model_type": "structured_augmented",
        },
        model_out,
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


def _validation_score(model: StructuredCorrMapMLP, route_val_loader: DataLoader, intr_val_loader: DataLoader) -> float:
    route_acc = _eval_route_accuracy(model, route_val_loader, 0.72, 0.5)
    intr_acc = _eval_intrinsic_accuracy(model, intr_val_loader)
    return 0.7 * route_acc + 0.3 * intr_acc


def _select_thresholds(*, model: StructuredCorrMapMLP, feature_mean: torch.Tensor, feature_std: torch.Tensor, benchmark_6x6: list[dict], temp_model_path: Path, ood_distance_threshold: float) -> tuple[float, float, float]:
    candidates_mf = [0.7, 0.8]
    candidates_q = [0.45, 0.55]
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
            "ood_distance_threshold": ood_distance_threshold,
            "model_type": "structured_augmented",
        },
        temp_model_path,
    )
    best = (0.75, 0.5)
    best_score = float("-inf")
    for mf_t in candidates_mf:
        for q_t in candidates_q:
            checkpoint = torch.load(temp_model_path, map_location="cpu")
            checkpoint["mean_field_threshold"] = mf_t
            checkpoint["quantum_threshold"] = q_t
            torch.save(checkpoint, temp_model_path)
            report = evaluate_regime_benchmark_structured(samples=benchmark_6x6, model_path=temp_model_path)
            summary = report["summary"]
            mf_f1 = float(summary["per_class_metrics"].get("mean_field", {}).get("f1", 0.0))
            sc_f1 = float(summary["per_class_metrics"].get("scalable_classical", {}).get("f1", 0.0))
            q_f1 = float(summary["per_class_metrics"].get("quantum_frontier", {}).get("f1", 0.0))
            tfim_acc = float(summary["by_family"].get("tfim", {}).get("accuracy", 0.0))
            score = float(summary["overall_accuracy"]) + 0.25 * mf_f1 + 0.25 * sc_f1 + 0.15 * q_f1 + 0.35 * tfim_acc
            if score > best_score:
                best = (mf_t, q_t)
                best_score = score
    temp_model_path.unlink(missing_ok=True)
    return best[0], best[1], best_score


def _eval_route_accuracy(model: StructuredCorrMapMLP, loader: DataLoader, mean_field_threshold: float, quantum_threshold: float) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for features, mf_targets, q_targets, route_labels in loader:
            out = model(features)
            preds = _decode_batch(out["mean_field_logit"], out["quantum_frontier_logit"], mean_field_threshold, quantum_threshold)
            truths = route_labels.tolist()
            values.append(sum(int(p == t) for p, t in zip(preds, truths, strict=True)) / max(len(truths), 1))
    return float(sum(values) / len(values)) if values else 0.0


def _eval_intrinsic_accuracy(model: StructuredCorrMapMLP, loader: DataLoader) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for features, labels, _, _ in loader:
            out = model(features)
            values.append(float((torch.argmax(out["intrinsic_logits"], dim=-1) == labels).float().mean().item()))
    return float(sum(values) / len(values)) if values else 0.0


def _decode_batch(mean_field_logit: torch.Tensor, quantum_logit: torch.Tensor, mean_field_threshold: float, quantum_threshold: float) -> list[int]:
    mf_probs = torch.sigmoid(mean_field_logit)
    q_probs = torch.sigmoid(quantum_logit)
    labels: list[int] = []
    for mf, q in zip(mf_probs.tolist(), q_probs.tolist(), strict=True):
        if q >= quantum_threshold:
            labels.append(ROUTING_LABELS.index("quantum_frontier"))
        elif mf >= mean_field_threshold:
            labels.append(ROUTING_LABELS.index("mean_field"))
        else:
            labels.append(ROUTING_LABELS.index("scalable_classical"))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dataset", type=Path, default=Path("backend/artifacts/routing_dataset_test.pt"))
    parser.add_argument("--intrinsic-dataset", type=Path, default=Path("backend/artifacts/intrinsic_corrmap_dataset_test.pt"))
    parser.add_argument("--benchmark-dataset", type=Path, default=Path("backend/artifacts/regime_benchmark.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/structured_corrmap_augmented.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/structured_corrmap_augmented.json"))
    parser.add_argument("--epochs", type=int, default=120)
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

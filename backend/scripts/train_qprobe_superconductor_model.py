from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.qprobe_superconductor_features import qprobe_superconductor_feature_dim
from app.ml.qprobe_superconductor_model import SuperconductorQProbeMLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("backend/artifacts/qprobe_superconductor_ml_dataset.pt"))
    parser.add_argument("--model-out", type=Path, default=Path("backend/artifacts/qprobe_superconductor_model.pt"))
    parser.add_argument("--metrics-out", type=Path, default=Path("backend/artifacts/qprobe_superconductor_model.json"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train(args.dataset, args.model_out, args.metrics_out, args.epochs, args.batch_size, args.learning_rate)
    print(json.dumps(metrics, indent=2))


def _split(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train, val, test = [], [], []
    for sample in samples:
        key = sample["metadata"]["problem_key"]
        if key == "U=8.0|mu=2.0":
            test.append(sample)
        elif key == "U=6.0|mu=1.5":
            val.append(sample)
        else:
            train.append(sample)
    if not test:
        # Quick datasets only include up to U=6.0.
        test = val
        val = []
    if not val:
        positives: dict[str, dict] = {}
        negatives: dict[str, dict] = {}
        remaining: list[dict] = []
        for sample in train:
            bundle = sample["metadata"]["bundle_name"]
            if sample["safe"] and bundle not in positives:
                positives[bundle] = sample
            elif (not sample["safe"]) and bundle not in negatives:
                negatives[bundle] = sample
            else:
                remaining.append(sample)
        val = list(positives.values()) + list(negatives.values())
        train = remaining
    return train, val, test


def _normalize(samples: list[dict], mean: torch.Tensor, std: torch.Tensor) -> list[dict]:
    return [{**sample, "features": (sample["features"] - mean) / std} for sample in samples]


def _loader(samples: list[dict], *, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.stack([sample["features"] for sample in samples], dim=0)
    y = torch.tensor([1 if sample["safe"] else 0 for sample in samples], dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: SuperconductorQProbeMLP, samples: list[dict], threshold: float) -> dict[str, float]:
    if not samples:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_rate": 0.0}
    loader = _loader(samples, batch_size=256, shuffle=False)
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            probs.extend(torch.sigmoid(model(x)).tolist())
            labels.extend(y.tolist())
    preds = [1 if p >= threshold else 0 for p in probs]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": sum(preds) / max(len(preds), 1),
    }


def train(dataset_path: Path, model_out: Path, metrics_out: Path, epochs: int, batch_size: int, learning_rate: float) -> dict[str, object]:
    samples = torch.load(dataset_path, map_location="cpu")
    train_samples, val_samples, test_samples = _split(samples)
    feature_mean = torch.stack([s["features"] for s in train_samples], dim=0).mean(dim=0)
    feature_std = torch.stack([s["features"] for s in train_samples], dim=0).std(dim=0).clamp_min(1e-6)
    train_samples = _normalize(train_samples, feature_mean, feature_std)
    val_samples = _normalize(val_samples, feature_mean, feature_std)
    test_samples = _normalize(test_samples, feature_mean, feature_std)

    model = SuperconductorQProbeMLP(input_dim=qprobe_superconductor_feature_dim())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos = sum(1 for s in train_samples if s["safe"])
    neg = max(len(train_samples) - pos, 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1), dtype=torch.float32))

    train_loader = _loader(train_samples, batch_size=batch_size, shuffle=True)
    best_state = None
    best_f1 = -1.0
    best_threshold = 0.5
    stale = 0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        for threshold in thresholds:
            metrics = _evaluate(model, val_samples, threshold)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        if stale > 25:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "threshold": best_threshold,
        },
        model_out,
    )
    metrics = {
        "threshold": best_threshold,
        "train": _evaluate(model, train_samples, best_threshold),
        "val": _evaluate(model, val_samples, best_threshold),
        "test": _evaluate(model, test_samples, best_threshold),
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
    }
    metrics_out.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()

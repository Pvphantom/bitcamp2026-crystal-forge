from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from app.ml.model import HubbardPhaseGraphNet
from app.ml.schema import (
    ARTIFACTS_DIR,
    DEFAULT_2X2_BASE_DATASET,
    DEFAULT_2X3_BASE_DATASET,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_DATASET,
    DEFAULT_TRAIN_DATASET,
    PHASE_LABELS,
    collate_graph_samples,
)
from app.ml.data import augment_samples


def evaluate(model: HubbardPhaseGraphNet, loader: DataLoader) -> tuple[float, list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                nodes=batch["nodes"],
                adjacency=batch["adjacency"],
                edge_attr=batch["edge_attr"],
                node_mask=batch["node_mask"],
                global_feats=batch["global_feats"],
            )
            preds.extend(torch.argmax(logits, dim=-1).tolist())
            labels.extend(batch["labels"].tolist())
    return accuracy_score(labels, preds), labels, preds


def split_train_val_by_base_id(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    base_labels = {}
    for sample in samples:
        base_labels[sample["metadata"]["base_id"]] = sample["label"]

    base_ids = sorted(base_labels.keys())
    val_base_ids = set(
        train_test_split(
            base_ids,
            test_size=0.15,
            random_state=11,
            stratify=[base_labels[base_id] for base_id in base_ids],
        )[1]
    )
    val_samples = [sample for sample in samples if sample["metadata"]["base_id"] in val_base_ids]
    train_samples = [sample for sample in samples if sample["metadata"]["base_id"] not in val_base_ids]
    return train_samples, val_samples


def train_model(
    *,
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[HubbardPhaseGraphNet, dict]:
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_samples)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_samples)
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_samples)

    class_counts = Counter(sample["label"] for sample in train_samples)
    weights = torch.tensor(
        [1.0 / max(class_counts.get(index, 1), 1) for index in range(len(PHASE_LABELS))],
        dtype=torch.float32,
    )
    weights = weights / weights.sum() * len(PHASE_LABELS)

    model = HubbardPhaseGraphNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_state = None
    best_val_acc = -1.0
    patience = 30
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                nodes=batch["nodes"],
                adjacency=batch["adjacency"],
                edge_attr=batch["edge_attr"],
                node_mask=batch["node_mask"],
                global_feats=batch["global_feats"],
            )
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()

        val_acc, _, _ = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    train_acc, _, _ = evaluate(model, train_loader)
    val_acc, _, _ = evaluate(model, val_loader)
    test_acc, test_labels, test_preds = evaluate(model, test_loader)

    confusion = confusion_matrix(test_labels, test_preds, labels=list(range(len(PHASE_LABELS)))).tolist()
    metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "class_counts_train": dict(class_counts),
        "confusion_matrix": confusion,
        "labels": PHASE_LABELS,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
    }
    return model, metrics


def train(
    *,
    train_path: Path,
    test_path: Path,
    model_out: Path,
    metrics_out: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    two_by_two_base_path: Path,
    two_by_three_base_path: Path,
) -> None:
    train_samples = torch.load(train_path, map_location="cpu")
    test_samples = torch.load(test_path, map_location="cpu")
    train_samples, val_samples = split_train_val_by_base_id(train_samples)
    model, metrics = train_model(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    base_2x2 = torch.load(two_by_two_base_path, map_location="cpu")
    base_2x3 = torch.load(two_by_three_base_path, map_location="cpu")
    train_2x2 = augment_samples(base_2x2)
    train_2x2, val_2x2 = split_train_val_by_base_id(train_2x2)
    cross_model, cross_metrics = train_model(
        train_samples=train_2x2,
        val_samples=val_2x2,
        test_samples=base_2x3,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    metrics["cross_lattice_accuracy"] = cross_metrics["test_accuracy"]
    metrics["cross_lattice_num_train_samples"] = cross_metrics["num_train_samples"]
    metrics["cross_lattice_num_test_samples"] = cross_metrics["num_test_samples"]
    metrics["cross_lattice_confusion_matrix"] = cross_metrics["confusion_matrix"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": {
                "node_in": 5,
                "edge_in": 1,
                "global_in": 3,
                "hidden_dim": 64,
                "num_layers": 3,
                "n_classes": len(PHASE_LABELS),
            },
            "labels": PHASE_LABELS,
        },
        model_out,
    )
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in cross_model.state_dict().items()},
            "model_config": {
                "node_in": 5,
                "edge_in": 1,
                "global_in": 3,
                "hidden_dim": 64,
                "num_layers": 3,
                "n_classes": len(PHASE_LABELS),
            },
            "labels": PHASE_LABELS,
        },
        ARTIFACTS_DIR / "phase_gnn_2x2_only.pt",
    )
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--two-by-two-base", type=Path, default=DEFAULT_2X2_BASE_DATASET)
    parser.add_argument("--two-by-three-base", type=Path, default=DEFAULT_2X3_BASE_DATASET)
    args = parser.parse_args()

    train(
        train_path=args.train_data,
        test_path=args.test_data,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        two_by_two_base_path=args.two_by_two_base,
        two_by_three_base_path=args.two_by_three_base,
    )


if __name__ == "__main__":
    main()

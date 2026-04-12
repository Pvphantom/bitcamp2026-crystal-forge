from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from app.ml.infer import RoutingInferenceEngine
from app.ml.schema import DEFAULT_ROUTING_DATASET, DEFAULT_ROUTING_METRICS_PATH, DEFAULT_ROUTING_MODEL_PATH


def evaluate_samples(
    samples: list[dict],
    *,
    inference: RoutingInferenceEngine,
) -> dict[str, Any]:
    rows = []
    for sample in samples:
        prediction = inference.predict(sample["features"])
        if prediction is None:
            continue
        truth = sample["route_label"]
        predicted = prediction["label"]
        abstained = bool(prediction.get("abstained", False))
        row = {
            "truth": truth,
            "predicted": predicted,
            "correct": truth == predicted,
            "covered": not abstained,
            "abstained": abstained,
            "abstain_reason": prediction.get("abstain_reason"),
            "reference_quality": sample.get("reference_quality", "unknown"),
            "family": sample.get("problem_metadata", {}).get("family", "unknown"),
            "lattice": _lattice_key(sample),
            "confidence": prediction.get("confidence"),
        }
        rows.append(row)
    return {
        "overall": _aggregate_rows(rows),
        "by_reference_quality": _group_metrics(rows, key="reference_quality"),
        "by_family": _group_metrics(rows, key="family"),
        "by_lattice": _group_metrics(rows, key="lattice"),
        "by_truth_label": _group_metrics(rows, key="truth"),
        "counts": {
            "num_samples": len(rows),
            "truth_label_counts": dict(Counter(row["truth"] for row in rows)),
            "predicted_label_counts": dict(Counter(row["predicted"] for row in rows)),
            "abstain_reason_counts": dict(Counter(row["abstain_reason"] for row in rows if row["abstained"])),
        },
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "route_accuracy": None,
            "covered_accuracy": None,
            "coverage": None,
            "abstention_rate": None,
            "ood_abstention_rate": None,
            "low_confidence_abstention_rate": None,
            "false_mean_field_rate": None,
            "regret_rate": None,
        }
    covered_rows = [row for row in rows if row["covered"]]
    abstained_rows = [row for row in rows if row["abstained"]]
    ood_rows = [row for row in abstained_rows if row["abstain_reason"] == "ood_distance"]
    low_conf_rows = [row for row in abstained_rows if row["abstain_reason"] == "low_confidence"]
    correct_rows = [row for row in rows if row["correct"]]
    covered_correct = [row for row in covered_rows if row["correct"]]
    mean_field_predictions = [row for row in covered_rows if row["predicted"] == "mean_field"]
    false_mean_field = [row for row in mean_field_predictions if row["truth"] != "mean_field"]
    regret_rows = [row for row in covered_rows if row["predicted"] != row["truth"]]
    return {
        "route_accuracy": len(correct_rows) / total,
        "covered_accuracy": None if not covered_rows else len(covered_correct) / len(covered_rows),
        "coverage": len(covered_rows) / total,
        "abstention_rate": len(abstained_rows) / total,
        "ood_abstention_rate": len(ood_rows) / total,
        "low_confidence_abstention_rate": len(low_conf_rows) / total,
        "false_mean_field_rate": None if not mean_field_predictions else len(false_mean_field) / len(mean_field_predictions),
        "regret_rate": None if not covered_rows else len(regret_rows) / len(covered_rows),
    }


def _group_metrics(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        group_key: {
            **_aggregate_rows(group_rows),
            "num_samples": len(group_rows),
        }
        for group_key, group_rows in sorted(grouped.items())
    }


def _lattice_key(sample: dict) -> str:
    metadata = sample.get("problem_metadata", {})
    Lx = metadata.get("Lx")
    Ly = metadata.get("Ly")
    if Lx is None or Ly is None:
        return "unknown"
    return f"{Lx}x{Ly}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROUTING_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_ROUTING_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROUTING_METRICS_PATH.parent / "routing_eval.json")
    args = parser.parse_args()

    samples = torch.load(args.dataset, map_location="cpu")
    inference = RoutingInferenceEngine(model_path=args.model)
    metrics = evaluate_samples(samples, inference=inference)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

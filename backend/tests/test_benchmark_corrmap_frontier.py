from pathlib import Path

import torch

import scripts.benchmark_corrmap_frontier as frontier


def test_filter_small_lattice_training_samples_excludes_large_lattices() -> None:
    samples = [
        {"problem_metadata": {"nsites": 4}},
        {"problem_metadata": {"nsites": 6}},
        {"problem_metadata": {"nsites": 16}},
        {"problem_metadata": {"nsites": 36}},
    ]
    filtered = frontier.filter_small_lattice_training_samples(samples, max_train_nsites=6)
    assert [sample["problem_metadata"]["nsites"] for sample in filtered] == [4, 6]


def test_frontier_summary_tracks_abstentions_and_route_counts() -> None:
    rows = [
        {"model_family": "hubbard", "lattice": "4x4", "route_label": "mean_field", "abstained": False, "confidence": 0.8, "intrinsic_label": "stable_classical"},
        {"model_family": "hubbard", "lattice": "4x4", "route_label": "uncertain", "abstained": True, "confidence": 0.4, "intrinsic_label": "frontier_or_uncertain", "abstain_reason": "intrinsic_risk_guard"},
        {"model_family": "tfim", "lattice": "6x6", "route_label": "quantum_frontier", "abstained": False, "confidence": 0.9, "intrinsic_label": "fragile_classical"},
    ]
    summary = frontier.summarize_frontier_predictions(rows)
    assert summary["overall"]["num_samples"] == 3
    assert summary["overall"]["route_label_counts"]["mean_field"] == 1
    assert summary["overall"]["route_label_counts"]["uncertain"] == 1
    assert summary["overall"]["intrinsic_label_counts"]["frontier_or_uncertain"] == 1
    assert summary["overall"]["intrinsic_guard_rate"] == 1 / 3
    assert summary["by_family"]["hubbard"]["num_samples"] == 2
    assert summary["by_lattice"]["6x6"]["route_label_counts"]["quantum_frontier"] == 1
    assert summary["by_intrinsic_label"]["stable_classical"]["num_samples"] == 1


def test_orchestrate_frontier_benchmark_never_trains_on_large_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_dataset_path = tmp_path / "train.pt"
    model_out = tmp_path / "model.pt"
    train_metrics_out = tmp_path / "train_metrics.json"
    benchmark_report_out = tmp_path / "benchmark.json"

    raw_samples = [
        {"features": torch.zeros(22), "route_label": "mean_field", "reference_quality": "strong", "problem_metadata": {"nsites": 4}},
        {"features": torch.ones(22), "route_label": "scalable_classical", "reference_quality": "strong", "problem_metadata": {"nsites": 16}},
    ]
    torch.save(raw_samples, train_dataset_path)

    captured = {}

    def fake_train(**kwargs):
        filtered = torch.load(kwargs["dataset_path"], map_location="cpu")
        captured["trained_nsites"] = [sample["problem_metadata"]["nsites"] for sample in filtered]
        model_out.write_bytes(b"stub")
        train_metrics_out.write_text("{}")
        return {"route_label_counts": {"mean_field": 1}}

    def fake_build_frontier_rows():
        return [
            {
                "sample_id": 1,
                "model_family": "hubbard",
                "lattice": "4x4",
                "nsites": 16,
                "parameters": {"U": 4.0, "mu": 2.0},
                "cheap_solver": "mean_field",
                "cheap_energy": -1.0,
                "cheap_observables": {"D": 0.2},
                "features": torch.zeros(22),
            }
        ]

    def fake_run_frontier_predictions(rows, *, inference):
        return {
            "summary": {"overall": {"num_samples": len(rows), "route_label_counts": {"mean_field": 1}}},
            "rows": [{"sample_id": rows[0]["sample_id"], "route_label": "mean_field", "abstained": False}],
        }

    monkeypatch.setattr(frontier, "train", fake_train)
    monkeypatch.setattr(frontier, "build_frontier_prediction_rows", fake_build_frontier_rows)
    monkeypatch.setattr(frontier, "run_frontier_predictions", fake_run_frontier_predictions)

    report = frontier.orchestrate_frontier_benchmark(
        train_dataset_path=train_dataset_path,
        model_out=model_out,
        train_metrics_out=train_metrics_out,
        benchmark_report_out=benchmark_report_out,
        max_train_nsites=6,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
    )

    assert captured["trained_nsites"] == [4]
    assert report["training_policy"]["large_lattice_training_labels_used"] == 0
    assert report["training_policy"]["filtered_out_large_training_samples"] == 1

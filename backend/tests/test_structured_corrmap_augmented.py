from pathlib import Path

import torch

from scripts.train_structured_corrmap_augmented import train


def test_structured_augmented_training_keeps_8x8_held_out(tmp_path: Path) -> None:
    routing_dataset = tmp_path / "routing.pt"
    intrinsic_dataset = tmp_path / "intrinsic.pt"
    benchmark_dataset = tmp_path / "benchmark.pt"
    model_path = tmp_path / "structured.pt"
    metrics_path = tmp_path / "structured.json"

    routing_samples = []
    for idx in range(4):
        routing_samples.extend(
            [
                {"features": torch.zeros(22) + 0.05 * idx, "route_label": "mean_field", "reference_quality": "strong", "problem_metadata": {"Lx": 2, "Ly": 2}},
                {"features": torch.ones(22) + 0.05 * idx, "route_label": "scalable_classical", "reference_quality": "strong", "problem_metadata": {"Lx": 2, "Ly": 3}},
                {"features": -torch.ones(22) + 0.05 * idx, "route_label": "quantum_frontier", "reference_quality": "strong", "problem_metadata": {"Lx": 2, "Ly": 3}},
            ]
        )

    intrinsic_base = {
        "features": torch.zeros(22),
        "stability": {"converged_fraction": 1.0, "energy_std": 0.0, "energy_span": 0.0, "residual_mean": 0.0, "residual_max": 0.0, "final_delta_mean": 0.0, "distinct_solution_count": 1, "observable_spans": {"x": 0.0}},
        "sensitivity": {"energy_density_shift_max": 0.0, "observable_shift_max": 0.0, "observable_shift_by_param": {"p": 0.0}},
        "size_consistency": {"observable_shift_max": 0.0, "energy_density_shift": 0.0, "observable_shift_by_name": {"x": 0.0}},
        "ansatz_disagreement": {"max_abs_gap": 0.0, "observable_gap_norm": 0.0, "energy_density_gap": 0.0},
        "hysteresis": {"observable_gap_max": 0.0, "energy_density_gap": 0.0},
    }
    intrinsic_samples = []
    for _ in range(4):
        intrinsic_samples.extend(
            [
                {**intrinsic_base, "intrinsic_label": "stable_classical", "problem_metadata": {"Lx": 4, "Ly": 4}},
                {**intrinsic_base, "intrinsic_label": "fragile_classical", "problem_metadata": {"Lx": 6, "Ly": 6}},
                {**intrinsic_base, "intrinsic_label": "frontier_or_uncertain", "problem_metadata": {"Lx": 6, "Ly": 6}},
            ]
        )

    benchmark_samples = [
        {"sample_id": 1, "benchmark_label": "mean_field", "rationale": "test", "problem": {"model_family": "hubbard", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"t": 1.0, "U": 1.0, "mu": 0.0}}},
        {"sample_id": 2, "benchmark_label": "scalable_classical", "rationale": "test", "problem": {"model_family": "hubbard", "Lx": 8, "Ly": 8, "boundary": "open", "parameters": {"t": 1.0, "U": 4.0, "mu": 2.0}}},
    ]
    torch.save(routing_samples, routing_dataset)
    torch.save(intrinsic_samples, intrinsic_dataset)
    torch.save(benchmark_samples, benchmark_dataset)

    metrics = train(
        routing_dataset_path=routing_dataset,
        intrinsic_dataset_path=intrinsic_dataset,
        benchmark_dataset_path=benchmark_dataset,
        model_out=model_path,
        metrics_out=metrics_path,
        epochs=5,
        batch_size=2,
        learning_rate=1e-3,
    )
    assert model_path.exists()
    assert metrics["training_lattices"]["routing"] == ["2x2", "2x3"]
    assert metrics["training_lattices"]["intrinsic"] == ["4x4", "6x6"]
    assert metrics["training_lattices"]["benchmark_held_out"] == ["8x8"]

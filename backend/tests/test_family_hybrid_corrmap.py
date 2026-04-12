from pathlib import Path

import torch

from scripts.eval_corrmap_family_models_on_regime_benchmark import evaluate_regime_benchmark_family_models
from scripts.train_hybrid_corrmap_augmented import train


def test_family_filtered_augmented_training_runs(tmp_path: Path) -> None:
    routing_dataset = tmp_path / "routing.pt"
    intrinsic_dataset = tmp_path / "intrinsic.pt"
    benchmark_dataset = tmp_path / "benchmark.pt"
    model_path = tmp_path / "hubbard.pt"
    metrics_path = tmp_path / "hubbard.json"

    routing_samples = [
        {"features": torch.zeros(22), "route_label": "mean_field", "reference_quality": "strong", "problem_metadata": {"family": "hubbard", "Lx": 2, "Ly": 2}},
        {"features": torch.ones(22), "route_label": "scalable_classical", "reference_quality": "strong", "problem_metadata": {"family": "hubbard", "Lx": 2, "Ly": 2}},
        {"features": 2 * torch.ones(22), "route_label": "mean_field", "reference_quality": "strong", "problem_metadata": {"family": "tfim", "Lx": 2, "Ly": 2}},
        {"features": 3 * torch.ones(22), "route_label": "quantum_frontier", "reference_quality": "strong", "problem_metadata": {"family": "tfim", "Lx": 2, "Ly": 2}},
    ] * 4
    intrinsic_base = {
        "stability": {"converged_fraction": 1.0, "energy_std": 0.0, "energy_span": 0.0, "residual_mean": 0.0, "residual_max": 0.0, "final_delta_mean": 0.0, "distinct_solution_count": 1, "observable_spans": {"x": 0.0}},
        "sensitivity": {"energy_density_shift_max": 0.0, "observable_shift_max": 0.0, "observable_shift_by_param": {"p": 0.0}},
        "size_consistency": {"observable_shift_max": 0.0, "energy_density_shift": 0.0, "observable_shift_by_name": {"x": 0.0}},
        "ansatz_disagreement": {"max_abs_gap": 0.0, "observable_gap_norm": 0.0, "energy_density_gap": 0.0},
        "hysteresis": {"observable_gap_max": 0.0, "energy_density_gap": 0.0},
    }
    intrinsic_samples = [
        {"features": torch.zeros(22), "intrinsic_label": "stable_classical", "problem_metadata": {"family": "hubbard", "Lx": 6, "Ly": 6}, **intrinsic_base},
        {"features": torch.ones(22), "intrinsic_label": "fragile_classical", "problem_metadata": {"family": "hubbard", "Lx": 6, "Ly": 6}, **intrinsic_base},
        {"features": 2 * torch.ones(22), "intrinsic_label": "frontier_or_uncertain", "problem_metadata": {"family": "tfim", "Lx": 6, "Ly": 6}, **intrinsic_base},
        {"features": 3 * torch.ones(22), "intrinsic_label": "stable_classical", "problem_metadata": {"family": "tfim", "Lx": 6, "Ly": 6}, **intrinsic_base},
    ] * 4
    benchmark_samples = [
        {"sample_id": 1, "benchmark_label": "mean_field", "rationale": "test", "problem": {"model_family": "hubbard", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"t": 1.0, "U": 1.0, "mu": 0.0}}},
        {"sample_id": 2, "benchmark_label": "scalable_classical", "rationale": "test", "problem": {"model_family": "hubbard", "Lx": 8, "Ly": 8, "boundary": "open", "parameters": {"t": 1.0, "U": 4.0, "mu": 2.0}}},
        {"sample_id": 3, "benchmark_label": "quantum_frontier", "rationale": "test", "problem": {"model_family": "tfim", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"J": 1.0, "h": 1.0, "g": 0.0}}},
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
        family_filter="hubbard",
    )
    assert model_path.exists()
    assert metrics["family_filter"] == "hubbard"


def test_family_model_benchmark_eval_runs(tmp_path: Path) -> None:
    samples = [
        {"sample_id": 1, "benchmark_label": "mean_field", "rationale": "test", "problem": {"model_family": "hubbard", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"t": 1.0, "U": 1.0, "mu": 0.0}}},
        {"sample_id": 2, "benchmark_label": "quantum_frontier", "rationale": "test", "problem": {"model_family": "tfim", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"J": 1.0, "h": 1.0, "g": 0.0}}},
    ]
    hubbard_model = Path("backend/artifacts/hybrid_corrmap_augmented.pt")
    tfim_model = Path("backend/artifacts/hybrid_corrmap_augmented.pt")
    if not hubbard_model.exists():
        return
    report = evaluate_regime_benchmark_family_models(
        samples=samples,
        hubbard_model_path=hubbard_model,
        tfim_model_path=tfim_model,
    )
    assert "summary" in report

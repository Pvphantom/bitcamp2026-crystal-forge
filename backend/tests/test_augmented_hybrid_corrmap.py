from pathlib import Path

import torch

from app.analysis.intrinsic_feature_vector import build_intrinsic_augmented_features
from scripts.train_hybrid_corrmap_augmented import train


def test_build_intrinsic_augmented_features_has_expected_dim() -> None:
    sample = {
        "features": torch.zeros(22, dtype=torch.float32),
        "stability": {
            "converged_fraction": 1.0,
            "energy_std": 0.0,
            "energy_span": 0.0,
            "residual_mean": 0.0,
            "residual_max": 0.0,
            "final_delta_mean": 0.0,
            "distinct_solution_count": 1,
            "observable_spans": {"x": 0.0},
        },
        "sensitivity": {
            "energy_density_shift_max": 0.0,
            "observable_shift_max": 0.0,
            "observable_shift_by_param": {"p": 0.0},
        },
        "size_consistency": {
            "observable_shift_max": 0.0,
            "energy_density_shift": 0.0,
            "observable_shift_by_name": {"x": 0.0},
        },
        "ansatz_disagreement": {
            "max_abs_gap": 0.0,
            "observable_gap_norm": 0.0,
            "energy_density_gap": 0.0,
        },
        "hysteresis": {
            "observable_gap_max": 0.0,
            "energy_density_gap": 0.0,
        },
        "physical_tractability": {
            "mean_field_plausibility": 1.0,
            "scalable_classical_plausibility": 0.5,
            "quantum_frontier_pressure": 0.0,
            "sign_problem_risk": 0.0,
            "stoquasticity": 1.0,
            "factorization_proxy": 1.0,
            "interaction_pressure": 0.0,
            "critical_pressure": 0.0,
        },
    }
    features = build_intrinsic_augmented_features(sample)
    assert features.shape == (49,)


def test_augmented_hybrid_training_respects_feature_shapes(tmp_path: Path) -> None:
    routing_dataset = tmp_path / "routing.pt"
    intrinsic_dataset = tmp_path / "intrinsic.pt"
    benchmark_dataset = tmp_path / "benchmark.pt"
    model_path = tmp_path / "hybrid_aug.pt"
    metrics_path = tmp_path / "hybrid_aug.json"

    routing_samples = []
    for idx in range(8):
        routing_samples.append(
            {
                "features": torch.full((22,), 0.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "mean_field",
                "reference_quality": "strong",
                "problem_metadata": {"nsites": 4, "Lx": 2, "Ly": 2},
            }
        )
        routing_samples.append(
            {
                "features": torch.full((22,), 3.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "scalable_classical",
                "reference_quality": "strong",
                "problem_metadata": {"nsites": 4, "Lx": 2, "Ly": 2},
            }
        )
        routing_samples.append(
            {
                "features": torch.full((22,), -3.0 + 0.05 * idx, dtype=torch.float32),
                "route_label": "quantum_frontier",
                "reference_quality": "strong",
                "problem_metadata": {"nsites": 4, "Lx": 2, "Ly": 2},
            }
        )
    intrinsic_samples = []
    centers = {
        "stable_classical": 0.0,
        "fragile_classical": 2.0,
        "frontier_or_uncertain": 6.0,
    }
    for label, center in centers.items():
        for idx in range(8):
            intrinsic_samples.append(
                {
                    "features": torch.full((22,), center + 0.05 * idx, dtype=torch.float32),
                    "intrinsic_label": label,
                    "stability": {
                        "converged_fraction": 1.0 if label == "stable_classical" else 0.6,
                        "energy_std": center,
                        "energy_span": center,
                        "residual_mean": 0.1 * center,
                        "residual_max": 0.1 * center,
                        "final_delta_mean": 0.01 * center,
                        "distinct_solution_count": 1 + int(center > 0.0),
                        "observable_spans": {"x": 0.1 * center},
                    },
                    "sensitivity": {
                        "energy_density_shift_max": 0.05 * center,
                        "observable_shift_max": 0.05 * center,
                        "observable_shift_by_param": {"p": 0.05 * center},
                    },
                    "size_consistency": {
                        "observable_shift_max": 0.05 * center,
                        "energy_density_shift": 0.05 * center,
                        "observable_shift_by_name": {"x": 0.05 * center},
                    },
                    "ansatz_disagreement": {
                        "max_abs_gap": 0.05 * center,
                        "observable_gap_norm": 0.05 * center,
                        "energy_density_gap": 0.05 * center,
                    },
                    "hysteresis": {
                        "observable_gap_max": 0.02 * center,
                        "energy_density_gap": 0.01 * center,
                    },
                    "physical_tractability": {
                        "mean_field_plausibility": max(0.0, 1.0 - 0.1 * center),
                        "scalable_classical_plausibility": max(0.0, 0.8 - 0.05 * center),
                        "quantum_frontier_pressure": min(1.0, 0.1 * center),
                        "sign_problem_risk": min(1.0, 0.05 * center),
                        "stoquasticity": 1.0,
                        "factorization_proxy": max(0.0, 1.0 - 0.1 * center),
                        "interaction_pressure": min(1.0, 0.1 * center),
                        "critical_pressure": min(1.0, 0.1 * center),
                    },
                    "problem_metadata": {"Lx": 6, "Ly": 6},
                }
            )
    benchmark_samples = [
        {
            "sample_id": 1,
            "benchmark_label": "mean_field",
            "rationale": "test",
            "problem": {"model_family": "hubbard", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"t": 1.0, "U": 1.0, "mu": 0.0}},
        },
        {
            "sample_id": 2,
            "benchmark_label": "quantum_frontier",
            "rationale": "test",
            "problem": {"model_family": "tfim", "Lx": 6, "Ly": 6, "boundary": "open", "parameters": {"J": 1.0, "h": 1.0, "g": 0.0}},
        },
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
        epochs=20,
        batch_size=4,
        learning_rate=1e-3,
    )
    assert model_path.exists()
    assert metrics["num_routing_samples"] == len(routing_samples)
    assert metrics["num_intrinsic_samples"] == len(intrinsic_samples)

from pathlib import Path

import torch

from app.analysis.general_tractability_features import analyze_general_tractability_features
from app.analysis.intrinsic_feature_vector_general import (
    GENERAL_INTRINSIC_FEATURE_DIM,
    build_general_intrinsic_features,
)
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap
from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from scripts.train_hybrid_corrmap_general import train


def test_general_intrinsic_feature_vector_has_expected_dim() -> None:
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
        "general_tractability": {
            "spatial_uniformity": 1.0,
            "site_dispersion": 0.0,
            "bond_dispersion": 0.0,
            "bond_sign_conflict": 0.0,
            "continuation_curvature": 0.0,
            "fixed_point_stiffness": 1.0,
            "metastability_pressure": 0.0,
            "scale_transfer_stability": 1.0,
            "response_linearity": 1.0,
            "correlation_load": 0.0,
            "classical_obstruction": 0.0,
            "factorization_reserve": 1.0,
        },
    }
    assert build_general_intrinsic_features(sample).shape == (GENERAL_INTRINSIC_FEATURE_DIM,)


def test_general_tractability_report_produces_bounded_scores() -> None:
    problem = ProblemSpec.hubbard(Lx=4, Ly=4, t=1.0, U=4.0, mu=2.0)
    cheap = MeanFieldSolver().solve(problem)
    runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
    report = analyze_general_tractability_features(
        cheap_result=cheap,
        stability=runtime.stability,
        sensitivity=runtime.sensitivity,
        size_consistency=runtime.size_consistency,
        ansatz_disagreement=runtime.ansatz_disagreement,
        hysteresis=runtime.hysteresis,
        physical_tractability=runtime.physical_tractability,
    )
    for value in report.__dict__.values():
        assert 0.0 <= float(value) <= 1.0


def test_general_hybrid_training_keeps_8x8_out_of_training(tmp_path: Path) -> None:
    routing_dataset = tmp_path / "routing.pt"
    intrinsic_dataset = tmp_path / "intrinsic_general.pt"
    benchmark_dataset = tmp_path / "benchmark.pt"
    model_path = tmp_path / "hybrid_general.pt"
    metrics_path = tmp_path / "hybrid_general.json"

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
        "physical_tractability": {"route_prior": "mean_field", "mean_field_plausibility": 1.0, "scalable_classical_plausibility": 0.5, "quantum_frontier_pressure": 0.0, "sign_problem_risk": 0.0, "stoquasticity": 1.0, "factorization_proxy": 1.0, "interaction_pressure": 0.0, "critical_pressure": 0.0},
        "general_tractability": {"spatial_uniformity": 1.0, "site_dispersion": 0.0, "bond_dispersion": 0.0, "bond_sign_conflict": 0.0, "continuation_curvature": 0.0, "fixed_point_stiffness": 1.0, "metastability_pressure": 0.0, "scale_transfer_stability": 1.0, "response_linearity": 1.0, "correlation_load": 0.0, "classical_obstruction": 0.0, "factorization_reserve": 1.0},
    }
    intrinsic_samples = []
    for _ in range(4):
        intrinsic_samples.extend(
            [
                {**intrinsic_base, "intrinsic_label": "stable_classical", "problem_metadata": {"Lx": 4, "Ly": 4}},
                {**intrinsic_base, "intrinsic_label": "fragile_classical", "problem_metadata": {"Lx": 6, "Ly": 6}},
                {**intrinsic_base, "intrinsic_label": "frontier_or_uncertain", "problem_metadata": {"Lx": 6, "Ly": 6}, "physical_tractability": {**intrinsic_base["physical_tractability"], "route_prior": "quantum_frontier", "quantum_frontier_pressure": 1.0}},
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
    assert metrics["training_lattices"]["intrinsic"] == ["4x4", "6x6"]
    assert metrics["training_lattices"]["benchmark_held_out"] == ["8x8"]

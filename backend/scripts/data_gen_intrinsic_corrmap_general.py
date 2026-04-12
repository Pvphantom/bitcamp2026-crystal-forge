from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.analysis.general_tractability_features import analyze_general_tractability_features
from scripts.data_gen_intrinsic_corrmap import GRIDS, _problems_for_grid, build_intrinsic_sample
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/intrinsic_corrmap_dataset_general.pt"))
    parser.add_argument("--num-seeds", type=int, default=6)
    parser.add_argument("--init-noise-scale", type=float, default=0.05)
    parser.add_argument("--perturbation-scale", type=float, default=0.05)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = []
    sample_id = 0
    for (family, Lx, Ly), grid in GRIDS.items():
        for problem in _problems_for_grid(family, Lx, Ly, grid):
            sample_id += 1
            dataset.append(
                build_intrinsic_sample_general(
                    problem,
                    sample_id=sample_id,
                    num_seeds=args.num_seeds,
                    init_noise_scale=args.init_noise_scale,
                    perturbation_scale=args.perturbation_scale,
                )
            )
    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} general intrinsic CorrMap samples to {args.output}")


def build_intrinsic_sample_general(
    problem: ProblemSpec,
    *,
    sample_id: int,
    num_seeds: int,
    init_noise_scale: float,
    perturbation_scale: float,
) -> dict:
    base_sample = build_intrinsic_sample(
        problem,
        sample_id=sample_id,
        num_seeds=num_seeds,
        init_noise_scale=init_noise_scale,
        perturbation_scale=perturbation_scale,
    )
    cheap_solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
    cheap_result = cheap_solver.solve(problem)
    runtime_report = analyze_runtime_intrinsic_corrmap(
        problem,
        cheap_result=cheap_result,
        num_seeds=num_seeds,
        init_noise_scale=init_noise_scale,
        perturbation_scale=perturbation_scale,
    )
    general = analyze_general_tractability_features(
        cheap_result=cheap_result,
        stability=runtime_report.stability,
        sensitivity=runtime_report.sensitivity,
        size_consistency=runtime_report.size_consistency,
        ansatz_disagreement=runtime_report.ansatz_disagreement,
        hysteresis=runtime_report.hysteresis,
        physical_tractability=runtime_report.physical_tractability,
    )
    return {
        **base_sample,
        "general_tractability": {
            "spatial_uniformity": general.spatial_uniformity,
            "site_dispersion": general.site_dispersion,
            "bond_dispersion": general.bond_dispersion,
            "bond_sign_conflict": general.bond_sign_conflict,
            "continuation_curvature": general.continuation_curvature,
            "fixed_point_stiffness": general.fixed_point_stiffness,
            "metastability_pressure": general.metastability_pressure,
            "scale_transfer_stability": general.scale_transfer_stability,
            "response_linearity": general.response_linearity,
            "correlation_load": general.correlation_load,
            "classical_obstruction": general.classical_obstruction,
            "factorization_reserve": general.factorization_reserve,
        },
        "label_source": "intrinsic_only_general",
    }


if __name__ == "__main__":
    main()

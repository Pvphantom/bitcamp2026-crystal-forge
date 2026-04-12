from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap
from app.analysis.trust_features import build_trust_feature_groups, flatten_trust_feature_groups
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


GRIDS = {
    ("hubbard", 4, 4): {"U": [1.0, 2.0, 4.0, 6.0, 8.0], "mu": [0.0, 2.0, 4.0]},
    ("hubbard", 6, 6): {"U": [1.0, 4.0, 8.0], "mu": [0.0, 2.0, 4.0]},
    ("tfim", 4, 4): {"J": [0.5, 1.0, 1.5], "h": [0.4, 1.0, 1.8], "g": [0.0, 0.5]},
    ("tfim", 6, 6): {"J": [1.0], "h": [0.4, 1.0, 1.8], "g": [0.0, 0.5]},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/intrinsic_corrmap_dataset.pt"))
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
                build_intrinsic_sample(
                    problem,
                    sample_id=sample_id,
                    num_seeds=args.num_seeds,
                    init_noise_scale=args.init_noise_scale,
                    perturbation_scale=args.perturbation_scale,
                )
            )
    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} intrinsic CorrMap samples to {args.output}")


def build_intrinsic_sample(
    problem: ProblemSpec,
    *,
    sample_id: int,
    num_seeds: int,
    init_noise_scale: float,
    perturbation_scale: float,
) -> dict:
    cheap_solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
    cheap_result = cheap_solver.solve(problem)
    runtime_report = analyze_runtime_intrinsic_corrmap(
        problem,
        cheap_result=cheap_result,
        num_seeds=num_seeds,
        init_noise_scale=init_noise_scale,
        perturbation_scale=perturbation_scale,
    )
    stability = runtime_report.stability
    sensitivity = runtime_report.sensitivity
    size_consistency = runtime_report.size_consistency
    ansatz_disagreement = runtime_report.ansatz_disagreement
    hysteresis = runtime_report.hysteresis
    physical = runtime_report.physical_tractability
    intrinsic = runtime_report.assessment
    feature_groups = build_trust_feature_groups(problem, cheap_result)
    return {
        "features": flatten_trust_feature_groups(feature_groups),
        "feature_groups": feature_groups,
        "intrinsic_label": intrinsic.label,
        "intrinsic_score": intrinsic.score,
        "intrinsic_reasons": intrinsic.reasons,
        "stability": {
            "num_runs": stability.num_runs,
            "converged_fraction": stability.converged_fraction,
            "energy_std": stability.energy_std,
            "energy_span": stability.energy_span,
            "observable_spans": stability.observable_spans,
            "residual_mean": stability.residual_mean,
            "residual_max": stability.residual_max,
            "final_delta_mean": stability.final_delta_mean,
            "distinct_solution_count": stability.distinct_solution_count,
        },
        "sensitivity": {
            "perturbation_scale": sensitivity.perturbation_scale,
            "energy_density_shift_max": sensitivity.energy_density_shift_max,
            "observable_shift_max": sensitivity.observable_shift_max,
            "observable_shift_by_param": sensitivity.observable_shift_by_param,
        },
        "size_consistency": {
            "reference_lattice": size_consistency.reference_lattice,
            "observable_shift_max": size_consistency.observable_shift_max,
            "energy_density_shift": size_consistency.energy_density_shift,
            "observable_shift_by_name": size_consistency.observable_shift_by_name,
        },
        "ansatz_disagreement": {
            "primary_solver": ansatz_disagreement.primary_solver,
            "alternate_solver": ansatz_disagreement.alternate_solver,
            "max_abs_gap": ansatz_disagreement.max_abs_gap,
            "observable_gap_norm": ansatz_disagreement.observable_gap_norm,
            "energy_density_gap": ansatz_disagreement.energy_density_gap,
            "risk_label": ansatz_disagreement.risk_label,
        },
        "hysteresis": {
            "control_parameter": hysteresis.control_parameter,
            "perturbation_scale": hysteresis.perturbation_scale,
            "lower_value": hysteresis.lower_value,
            "center_value": hysteresis.center_value,
            "upper_value": hysteresis.upper_value,
            "observable_gap_max": hysteresis.observable_gap_max,
            "energy_density_gap": hysteresis.energy_density_gap,
            "observable_gap_by_name": hysteresis.observable_gap_by_name,
        },
        "physical_tractability": {
            "mean_field_plausibility": physical.mean_field_plausibility,
            "scalable_classical_plausibility": physical.scalable_classical_plausibility,
            "quantum_frontier_pressure": physical.quantum_frontier_pressure,
            "sign_problem_risk": physical.sign_problem_risk,
            "stoquasticity": physical.stoquasticity,
            "factorization_proxy": physical.factorization_proxy,
            "interaction_pressure": physical.interaction_pressure,
            "critical_pressure": physical.critical_pressure,
            "route_prior": physical.route_prior,
            "reasons": list(physical.reasons),
        },
        "problem_metadata": {
            "sample_id": sample_id,
            "family": problem.model_family,
            "Lx": problem.Lx,
            "Ly": problem.Ly,
            "nsites": problem.nsites,
            "boundary": problem.lattice.boundary,
            **{key: float(value) for key, value in problem.parameters.values.items()},
        },
        "label_source": "intrinsic_only",
    }


def _problems_for_grid(family: str, Lx: int, Ly: int, grid: dict[str, list[float]]) -> list[ProblemSpec]:
    problems: list[ProblemSpec] = []
    if family == "hubbard":
        for U in grid["U"]:
            for mu in grid["mu"]:
                problems.append(ProblemSpec.hubbard(Lx=Lx, Ly=Ly, t=1.0, U=U, mu=mu))
        return problems
    if family == "tfim":
        for J in grid["J"]:
            for h in grid["h"]:
                for g in grid["g"]:
                    problems.append(ProblemSpec.tfim(Lx=Lx, Ly=Ly, J=J, h=h, g=g))
        return problems
    raise ValueError(f"Unsupported model family: {family}")


if __name__ == "__main__":
    main()

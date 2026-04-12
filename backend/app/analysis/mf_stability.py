from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSettings, MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSettings, TFIMMeanFieldSolver


@dataclass(frozen=True)
class MeanFieldStabilityReport:
    num_runs: int
    converged_fraction: float
    energy_std: float
    energy_span: float
    observable_spans: dict[str, float]
    residual_mean: float
    residual_max: float
    final_delta_mean: float
    distinct_solution_count: int


def analyze_mean_field_stability(
    problem: ProblemSpec,
    *,
    num_seeds: int = 6,
    init_noise_scale: float = 0.05,
    energy_cluster_tol: float = 1e-3,
) -> MeanFieldStabilityReport:
    results = []
    for seed in range(num_seeds):
        if problem.model_family == "hubbard":
            solver = MeanFieldSolver(MeanFieldSettings(seed=seed, init_noise_scale=init_noise_scale))
        elif problem.model_family == "tfim":
            solver = TFIMMeanFieldSolver(TFIMMeanFieldSettings(seed=seed, init_noise_scale=init_noise_scale))
        else:
            raise ValueError(f"Unsupported model family: {problem.model_family}")
        results.append(solver.solve(problem))

    energies = [float(result.energy) for result in results]
    residuals = [float(result.metadata.get("residual_norm", 0.0)) for result in results]
    final_deltas = [float(result.metadata.get("final_delta", 0.0)) for result in results]
    converged_fraction = mean(1.0 if bool(result.metadata.get("converged", False)) else 0.0 for result in results)
    observable_keys = [key for key in results[0].global_observables.keys() if key != "energy"]
    observable_spans = {
        key: max(float(result.global_observables[key]) for result in results)
        - min(float(result.global_observables[key]) for result in results)
        for key in observable_keys
    }
    rounded_energies = {round(energy / energy_cluster_tol) for energy in energies}
    return MeanFieldStabilityReport(
        num_runs=len(results),
        converged_fraction=float(converged_fraction),
        energy_std=float(pstdev(energies)) if len(energies) > 1 else 0.0,
        energy_span=float(max(energies) - min(energies)),
        observable_spans=observable_spans,
        residual_mean=float(mean(residuals)),
        residual_max=float(max(residuals)),
        final_delta_mean=float(mean(final_deltas)),
        distinct_solution_count=len(rounded_energies),
    )

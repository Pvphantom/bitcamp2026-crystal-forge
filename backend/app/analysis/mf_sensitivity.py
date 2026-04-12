from __future__ import annotations

from dataclasses import dataclass

from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


@dataclass(frozen=True)
class MeanFieldSensitivityReport:
    perturbation_scale: float
    energy_density_shift_max: float
    observable_shift_max: float
    observable_shift_by_param: dict[str, float]


def analyze_mean_field_sensitivity(
    problem: ProblemSpec,
    *,
    perturbation_scale: float = 0.05,
) -> MeanFieldSensitivityReport:
    base_solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
    base_result = base_solver.solve(problem)
    shifts_by_param: dict[str, float] = {}
    energy_density_shift_max = 0.0
    observable_shift_max = 0.0

    for param_name, value in problem.parameters.values.items():
        delta = perturbation_scale * max(abs(float(value)), 1.0)
        perturbed_problem = _perturb_problem(problem, param_name, float(value) + delta)
        solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
        perturbed = solver.solve(perturbed_problem)
        obs_shift = max(
            abs(float(perturbed.global_observables[key]) - float(base_result.global_observables[key]))
            for key in base_result.global_observables
            if key != "energy"
        )
        shifts_by_param[param_name] = float(obs_shift)
        observable_shift_max = max(observable_shift_max, float(obs_shift))
        energy_density_shift = abs(float(perturbed.energy) - float(base_result.energy)) / problem.nsites
        energy_density_shift_max = max(energy_density_shift_max, float(energy_density_shift))

    return MeanFieldSensitivityReport(
        perturbation_scale=perturbation_scale,
        energy_density_shift_max=float(energy_density_shift_max),
        observable_shift_max=float(observable_shift_max),
        observable_shift_by_param=shifts_by_param,
    )


def _perturb_problem(problem: ProblemSpec, param_name: str, param_value: float) -> ProblemSpec:
    params = dict(problem.parameters.values)
    params[param_name] = float(param_value)
    if problem.model_family == "hubbard":
        return ProblemSpec.hubbard(
            Lx=problem.Lx,
            Ly=problem.Ly,
            t=float(params["t"]),
            U=float(params["U"]),
            mu=float(params["mu"]),
            boundary=problem.lattice.boundary,
        )
    if problem.model_family == "tfim":
        return ProblemSpec.tfim(
            Lx=problem.Lx,
            Ly=problem.Ly,
            J=float(params["J"]),
            h=float(params["h"]),
            g=float(params.get("g", 0.0)),
            boundary=problem.lattice.boundary,
        )
    raise ValueError(f"Unsupported model family: {problem.model_family}")

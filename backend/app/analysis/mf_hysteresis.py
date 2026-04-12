from __future__ import annotations

from dataclasses import dataclass

from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSettings, MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSettings, TFIMMeanFieldSolver


@dataclass(frozen=True)
class MeanFieldHysteresisReport:
    control_parameter: str
    perturbation_scale: float
    lower_value: float
    center_value: float
    upper_value: float
    observable_gap_max: float
    energy_density_gap: float
    observable_gap_by_name: dict[str, float]


def analyze_mean_field_hysteresis(
    problem: ProblemSpec,
    *,
    perturbation_scale: float = 0.05,
) -> MeanFieldHysteresisReport:
    control_parameter = _control_parameter(problem)
    center_value = float(problem.parameters.values[control_parameter])
    delta = perturbation_scale * max(abs(center_value), 1.0)
    lower_value = center_value - delta
    upper_value = center_value + delta

    lower_problem = _perturb_problem(problem, control_parameter, lower_value)
    upper_problem = _perturb_problem(problem, control_parameter, upper_value)
    center_from_lower = _solve_center_from_neighbor(problem, lower_problem)
    center_from_upper = _solve_center_from_neighbor(problem, upper_problem)

    observable_keys = [
        key
        for key in center_from_lower.global_observables.keys()
        if key != "energy"
    ]
    gaps = {
        key: abs(float(center_from_lower.global_observables[key]) - float(center_from_upper.global_observables[key]))
        for key in observable_keys
    }
    return MeanFieldHysteresisReport(
        control_parameter=control_parameter,
        perturbation_scale=perturbation_scale,
        lower_value=float(lower_value),
        center_value=float(center_value),
        upper_value=float(upper_value),
        observable_gap_max=max(gaps.values(), default=0.0),
        energy_density_gap=abs(float(center_from_lower.energy) - float(center_from_upper.energy)) / max(problem.nsites, 1),
        observable_gap_by_name=gaps,
    )


def _control_parameter(problem: ProblemSpec) -> str:
    if problem.model_family == "hubbard":
        return "U"
    if problem.model_family == "tfim":
        return "h"
    raise ValueError(f"Unsupported model family: {problem.model_family}")


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


def _solve_center_from_neighbor(center_problem: ProblemSpec, neighbor_problem: ProblemSpec):
    if center_problem.model_family == "hubbard":
        neighbor = MeanFieldSolver().solve(neighbor_problem)
        solver = MeanFieldSolver(
            MeanFieldSettings(
                init_n_up=neighbor.site_observables["n_up"],
                init_n_dn=neighbor.site_observables["n_dn"],
            )
        )
        return solver.solve(center_problem)
    neighbor = TFIMMeanFieldSolver().solve(neighbor_problem)
    solver = TFIMMeanFieldSolver(
        TFIMMeanFieldSettings(
            init_mz=neighbor.site_observables["Mz_site"],
            init_mx=neighbor.site_observables["Mx_site"],
        )
    )
    return solver.solve(center_problem)

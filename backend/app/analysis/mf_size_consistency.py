from __future__ import annotations

from dataclasses import dataclass

from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


@dataclass(frozen=True)
class MeanFieldSizeConsistencyReport:
    reference_lattice: str | None
    observable_shift_max: float
    energy_density_shift: float
    observable_shift_by_name: dict[str, float]


def analyze_mean_field_size_consistency(problem: ProblemSpec) -> MeanFieldSizeConsistencyReport:
    reference_problem = _reference_problem(problem)
    if reference_problem is None:
        return MeanFieldSizeConsistencyReport(
            reference_lattice=None,
            observable_shift_max=0.0,
            energy_density_shift=0.0,
            observable_shift_by_name={},
        )

    solver = MeanFieldSolver() if problem.model_family == "hubbard" else TFIMMeanFieldSolver()
    current = solver.solve(problem)
    reference = solver.solve(reference_problem)
    observable_keys = [key for key in current.global_observables.keys() if key != "energy"]
    shifts = {
        key: abs(float(current.global_observables[key]) - float(reference.global_observables[key]))
        for key in observable_keys
    }
    return MeanFieldSizeConsistencyReport(
        reference_lattice=f"{reference_problem.Lx}x{reference_problem.Ly}",
        observable_shift_max=max(shifts.values(), default=0.0),
        energy_density_shift=abs(float(current.energy) / problem.nsites - float(reference.energy) / reference_problem.nsites),
        observable_shift_by_name=shifts,
    )


def _reference_problem(problem: ProblemSpec) -> ProblemSpec | None:
    if problem.Lx <= 2 or problem.Ly <= 2:
        return None
    ref_Lx = max(2, problem.Lx - 2)
    ref_Ly = max(2, problem.Ly - 2)
    if problem.model_family == "hubbard":
        return ProblemSpec.hubbard(
            Lx=ref_Lx,
            Ly=ref_Ly,
            t=problem.t,
            U=problem.U,
            mu=problem.mu,
            boundary=problem.lattice.boundary,
        )
    if problem.model_family == "tfim":
        return ProblemSpec.tfim(
            Lx=ref_Lx,
            Ly=ref_Ly,
            J=problem.J,
            h=problem.h,
            g=problem.g,
            boundary=problem.lattice.boundary,
        )
    raise ValueError(f"Unsupported model family: {problem.model_family}")

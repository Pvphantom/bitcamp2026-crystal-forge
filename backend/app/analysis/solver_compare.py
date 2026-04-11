from __future__ import annotations

from dataclasses import dataclass

from app.domain.problem_spec import ProblemSpec
from app.solvers.base import SolverResult


@dataclass(frozen=True)
class SolverComparison:
    problem: ProblemSpec
    reference_solver: str
    candidate_solver: str
    exact: dict[str, float]
    approximate: dict[str, float]
    abs_error: dict[str, float]
    rel_error: dict[str, float]
    max_abs_error: float
    observable_error_norm: float
    energy_error: float
    risk_label: str


def compare_solver_results(
    problem: ProblemSpec,
    reference: SolverResult,
    candidate: SolverResult,
    *,
    risk_thresholds: tuple[float, float] = (0.08, 0.2),
) -> SolverComparison:
    if problem.model_family == "hubbard":
        keys = ["D", "n", "Ms2", "K", "Cs_max"]
    elif problem.model_family == "tfim":
        keys = ["Mz", "Mx", "ZZ_nn", "Mstag2", "Z_span"]
    else:
        raise ValueError(f"Unsupported model family for solver comparison: {problem.model_family}")
    exact = {key: float(reference.global_observables[key]) for key in keys}
    approximate = {key: float(candidate.global_observables[key]) for key in keys}
    abs_error = {key: abs(approximate[key] - exact[key]) for key in keys}
    rel_error = {
        key: abs_error[key] / max(abs(exact[key]), 1e-8)
        for key in keys
    }
    max_abs_error = max(abs_error.values())
    observable_error_norm = float(sum(value * value for value in abs_error.values()) ** 0.5)
    energy_error = abs(candidate.energy - reference.energy)

    if max_abs_error < risk_thresholds[0] and energy_error < 0.25:
        risk = "safe"
    elif max_abs_error < risk_thresholds[1] and energy_error < 1.0:
        risk = "warning"
    else:
        risk = "unsafe"

    return SolverComparison(
        problem=problem,
        reference_solver=reference.solver_name,
        candidate_solver=candidate.solver_name,
        exact=exact,
        approximate=approximate,
        abs_error=abs_error,
        rel_error=rel_error,
        max_abs_error=max_abs_error,
        observable_error_norm=observable_error_norm,
        energy_error=energy_error,
        risk_label=risk,
    )

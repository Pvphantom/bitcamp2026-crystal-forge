from __future__ import annotations

from dataclasses import dataclass

from app.analysis.solver_compare import compare_solver_results
from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.paramagnetic_mean_field import ParamagneticMeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from app.solvers.uniform_tfim_mean_field import UniformTFIMMeanFieldSolver


@dataclass(frozen=True)
class MeanFieldAnsatzDisagreementReport:
    primary_solver: str
    alternate_solver: str
    max_abs_gap: float
    observable_gap_norm: float
    energy_density_gap: float
    risk_label: str


def analyze_mean_field_ansatz_disagreement(problem: ProblemSpec) -> MeanFieldAnsatzDisagreementReport:
    if problem.model_family == "hubbard":
        primary = MeanFieldSolver().solve(problem)
        alternate = ParamagneticMeanFieldSolver().solve(problem)
    elif problem.model_family == "tfim":
        primary = TFIMMeanFieldSolver().solve(problem)
        alternate = UniformTFIMMeanFieldSolver().solve(problem)
    else:
        raise ValueError(f"Unsupported model family for ansatz disagreement: {problem.model_family}")

    comparison = compare_solver_results(problem, reference=primary, candidate=alternate)
    return MeanFieldAnsatzDisagreementReport(
        primary_solver=primary.solver_name,
        alternate_solver=alternate.solver_name,
        max_abs_gap=float(comparison.max_abs_error),
        observable_gap_norm=float(comparison.observable_error_norm),
        energy_density_gap=float(comparison.energy_error / max(problem.nsites, 1)),
        risk_label=str(comparison.risk_label),
    )

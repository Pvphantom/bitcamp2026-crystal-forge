from __future__ import annotations

from dataclasses import dataclass
import math

from app.domain.problem_spec import ProblemSpec
from app.physics.lattice import nn_bonds
from app.solvers.base import BaseSolver, SolverResult


@dataclass(frozen=True)
class UniformTFIMMeanFieldSettings:
    max_iter: int = 200
    tol: float = 1e-8
    mixing: float = 0.4


class UniformTFIMMeanFieldSolver(BaseSolver):
    name = "uniform_tfim_mean_field"

    def __init__(self, settings: UniformTFIMMeanFieldSettings | None = None) -> None:
        self.settings = settings or UniformTFIMMeanFieldSettings()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family == "tfim"

    def solve(self, problem: ProblemSpec) -> SolverResult:
        coordination = 2.0 * len(nn_bonds(problem.Lx, problem.Ly)) / problem.nsites
        mz = 0.0
        mx = 0.5
        converged = False
        iterations = 0
        final_delta = float("inf")
        for iterations in range(1, self.settings.max_iter + 1):
            eff_z = problem.g + problem.J * coordination * mz
            norm = math.sqrt(problem.h * problem.h + eff_z * eff_z)
            if norm < 1e-12:
                next_mx = 0.0
                next_mz = 0.0
            else:
                next_mx = problem.h / norm
                next_mz = eff_z / norm
            mixed_mx = self.settings.mixing * next_mx + (1.0 - self.settings.mixing) * mx
            mixed_mz = self.settings.mixing * next_mz + (1.0 - self.settings.mixing) * mz
            final_delta = max(abs(mixed_mx - mx), abs(mixed_mz - mz))
            mx, mz = mixed_mx, mixed_mz
            if final_delta < self.settings.tol:
                converged = True
                break

        eff_z = problem.g + problem.J * coordination * mz
        norm = math.sqrt(problem.h * problem.h + eff_z * eff_z)
        residual_norm = 0.0 if norm < 1e-12 else max(abs(mx - problem.h / norm), abs(mz - eff_z / norm))
        zz_value = mz * mz
        energy = -problem.J * len(nn_bonds(problem.Lx, problem.Ly)) * zz_value - problem.h * problem.nsites * mx - problem.g * problem.nsites * mz
        mz_site = [float(mz)] * problem.nsites
        mx_site = [float(mx)] * problem.nsites
        return SolverResult(
            solver_name=self.name,
            energy=float(energy),
            global_observables={
                "Mz": float(mz),
                "Mx": float(mx),
                "ZZ_nn": float(zz_value),
                "Mstag2": 0.0,
                "Z_span": float(zz_value),
                "energy": float(energy),
            },
            site_observables={
                "Mz_site": mz_site,
                "Mx_site": mx_site,
            },
            bond_observables={(i, j): float(zz_value) for i, j in nn_bonds(problem.Lx, problem.Ly)},
            statevector=None,
            metadata={
                "converged": converged,
                "iterations": iterations,
                "final_delta": final_delta,
                "residual_norm": float(residual_norm),
                "ansatz": "uniform",
            },
        )

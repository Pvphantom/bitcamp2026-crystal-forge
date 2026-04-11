from __future__ import annotations

from dataclasses import dataclass
import math

from app.domain.problem_spec import ProblemSpec
from app.physics.lattice import nn_bonds
from app.solvers.base import BaseSolver, SolverResult


@dataclass(frozen=True)
class TFIMMeanFieldSettings:
    max_iter: int = 200
    tol: float = 1e-8
    mixing: float = 0.4


class TFIMMeanFieldSolver(BaseSolver):
    name = "tfim_mean_field"

    def __init__(self, settings: TFIMMeanFieldSettings | None = None) -> None:
        self.settings = settings or TFIMMeanFieldSettings()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family == "tfim"

    def solve(self, problem: ProblemSpec) -> SolverResult:
        nsites = problem.nsites
        mz = [0.1 if (i % 2 == 0) else -0.1 for i in range(nsites)]
        mx = [0.0 for _ in range(nsites)]
        neighbors = {i: [] for i in range(nsites)}
        for i, j in nn_bonds(problem.Lx, problem.Ly):
            neighbors[i].append(j)
            neighbors[j].append(i)

        converged = False
        iterations = 0
        for iterations in range(1, self.settings.max_iter + 1):
            next_mz = []
            next_mx = []
            for i in range(nsites):
                eff_z = problem.g + problem.J * sum(mz[j] for j in neighbors[i])
                norm = math.sqrt(problem.h * problem.h + eff_z * eff_z)
                if norm < 1e-12:
                    local_mx = 0.0
                    local_mz = 0.0
                else:
                    local_mx = problem.h / norm
                    local_mz = eff_z / norm
                next_mx.append(local_mx)
                next_mz.append(local_mz)
            mixed_mz = [
                self.settings.mixing * new + (1.0 - self.settings.mixing) * old
                for new, old in zip(next_mz, mz, strict=True)
            ]
            mixed_mx = [
                self.settings.mixing * new + (1.0 - self.settings.mixing) * old
                for new, old in zip(next_mx, mx, strict=True)
            ]
            delta = max(
                max(abs(a - b) for a, b in zip(mixed_mz, mz, strict=True)),
                max(abs(a - b) for a, b in zip(mixed_mx, mx, strict=True)),
            )
            mz = mixed_mz
            mx = mixed_mx
            if delta < self.settings.tol:
                converged = True
                break

        zz_bonds = {(i, j): mz[i] * mz[j] for i, j in nn_bonds(problem.Lx, problem.Ly)}
        avg_mz = sum(mz) / nsites
        avg_mx = sum(mx) / nsites
        avg_zz = sum(zz_bonds.values()) / max(len(zz_bonds), 1)
        staggered = 0.0
        for i, value in enumerate(mz):
            x = i % problem.Lx
            y = i // problem.Lx
            staggered += ((-1) ** (x + y)) * value / nsites
        mstag2 = staggered**2
        z_span = mz[0] * mz[-1]
        energy = (
            -problem.J * sum(zz_bonds.values())
            - problem.h * sum(mx)
            - problem.g * sum(mz)
        )
        return SolverResult(
            solver_name=self.name,
            energy=float(energy),
            global_observables={
                "Mz": float(avg_mz),
                "Mx": float(avg_mx),
                "ZZ_nn": float(avg_zz),
                "Mstag2": float(mstag2),
                "Z_span": float(z_span),
                "energy": float(energy),
            },
            site_observables={
                "Mz_site": list(map(float, mz)),
                "Mx_site": list(map(float, mx)),
            },
            bond_observables={bond: float(value) for bond, value in zz_bonds.items()},
            statevector=None,
            metadata={"converged": converged, "iterations": iterations},
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.domain.problem_spec import ProblemSpec
from app.physics.lattice import nn_bonds
from app.solvers.base import BaseSolver, SolverResult
from app.solvers.mean_field import _hopping_matrix, _occupy_negative_energy_states


@dataclass(frozen=True)
class ParamagneticMeanFieldSettings:
    max_iter: int = 200
    tol: float = 1e-8
    mixing: float = 0.4


class ParamagneticMeanFieldSolver(BaseSolver):
    name = "paramagnetic_mean_field"

    def __init__(self, settings: ParamagneticMeanFieldSettings | None = None) -> None:
        self.settings = settings or ParamagneticMeanFieldSettings()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family == "hubbard"

    def solve(self, problem: ProblemSpec) -> SolverResult:
        hopping = _hopping_matrix(problem)
        filling = 1.0
        converged = False
        iterations = 0
        final_delta = float("inf")

        for iterations in range(1, self.settings.max_iter + 1):
            h_eff = hopping + np.diag(np.full(problem.nsites, problem.U * filling / 2.0 - problem.mu, dtype=float))
            density, _ = _occupy_negative_energy_states(h_eff)
            new_filling = float(2.0 * np.sum(density) / problem.nsites)
            mixed_filling = self.settings.mixing * new_filling + (1.0 - self.settings.mixing) * filling
            final_delta = abs(mixed_filling - filling)
            filling = mixed_filling
            if final_delta < self.settings.tol:
                converged = True
                break

        h_eff = hopping + np.diag(np.full(problem.nsites, problem.U * filling / 2.0 - problem.mu, dtype=float))
        density, eigvals = _occupy_negative_energy_states(h_eff)
        residual_norm = abs(float(2.0 * np.sum(density) / problem.nsites) - filling)
        n_up_site = np.full(problem.nsites, filling / 2.0, dtype=float)
        n_dn_site = np.full(problem.nsites, filling / 2.0, dtype=float)
        d_site = np.full(problem.nsites, (filling / 2.0) ** 2, dtype=float)
        sz_site = np.zeros(problem.nsites, dtype=float)

        rho = np.zeros_like(hopping)
        vals, vecs = np.linalg.eigh(h_eff)
        for idx, energy in enumerate(vals):
            if energy < -1e-10:
                vec = vecs[:, idx]
                rho += np.outer(vec, vec.conj()).real
        nbonds = len(nn_bonds(problem.Lx, problem.Ly))
        kinetic = float(2.0 * np.trace(rho @ hopping).real / max(nbonds, 1))
        energy = float(2.0 * np.sum(vals[vals < -1e-10]) - problem.U * np.sum(d_site))

        return SolverResult(
            solver_name=self.name,
            energy=energy,
            global_observables={
                "D": float(np.mean(d_site)),
                "n": float(filling),
                "Ms2": 0.0,
                "K": kinetic,
                "Cs_max": 0.0,
                "energy": energy,
            },
            site_observables={
                "n_up": n_up_site.tolist(),
                "n_dn": n_dn_site.tolist(),
                "D_site": d_site.tolist(),
                "Sz_site": sz_site.tolist(),
            },
            bond_observables={(i, j): 0.0 for i, j in nn_bonds(problem.Lx, problem.Ly)},
            statevector=None,
            metadata={
                "converged": converged,
                "iterations": iterations,
                "final_delta": final_delta,
                "residual_norm": residual_norm,
                "ansatz": "paramagnetic_uniform",
            },
        )

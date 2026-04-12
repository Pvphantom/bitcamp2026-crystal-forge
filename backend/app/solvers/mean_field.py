from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.domain.problem_spec import ProblemSpec
from app.physics.lattice import nn_bonds
from app.solvers.base import BaseSolver, SolverResult


@dataclass(frozen=True)
class MeanFieldSettings:
    max_iter: int = 200
    tol: float = 1e-8
    mixing: float = 0.35
    seed: int | None = None
    init_noise_scale: float = 0.0
    init_n_up: Sequence[float] | None = None
    init_n_dn: Sequence[float] | None = None


def _hopping_matrix(problem: ProblemSpec) -> np.ndarray:
    matrix = np.zeros((problem.nsites, problem.nsites), dtype=float)
    for i, j in nn_bonds(problem.Lx, problem.Ly):
        matrix[i, j] = -problem.t
        matrix[j, i] = -problem.t
    return matrix


def _half_filled_af_seed(problem: ProblemSpec) -> tuple[np.ndarray, np.ndarray]:
    n_up = np.full(problem.nsites, 0.5, dtype=float)
    n_dn = np.full(problem.nsites, 0.5, dtype=float)
    for y in range(problem.Ly):
        for x in range(problem.Lx):
            i = x + problem.Lx * y
            stagger = 0.1 if (x + y) % 2 == 0 else -0.1
            n_up[i] += stagger
            n_dn[i] -= stagger
    return np.clip(n_up, 0.0, 1.0), np.clip(n_dn, 0.0, 1.0)


def _randomized_seed(problem: ProblemSpec, settings: MeanFieldSettings) -> tuple[np.ndarray, np.ndarray]:
    if settings.init_n_up is not None and settings.init_n_dn is not None:
        return (
            np.clip(np.asarray(settings.init_n_up, dtype=float), 0.0, 1.0),
            np.clip(np.asarray(settings.init_n_dn, dtype=float), 0.0, 1.0),
        )
    n_up, n_dn = _half_filled_af_seed(problem)
    if settings.init_noise_scale <= 0.0:
        return n_up, n_dn
    rng = np.random.default_rng(settings.seed)
    noise_up = rng.normal(scale=settings.init_noise_scale, size=problem.nsites)
    noise_dn = rng.normal(scale=settings.init_noise_scale, size=problem.nsites)
    return np.clip(n_up + noise_up, 0.0, 1.0), np.clip(n_dn + noise_dn, 0.0, 1.0)


def _occupy_negative_energy_states(ham: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(ham)
    occupied = eigvals < -1e-10
    density = np.sum(np.abs(eigvecs[:, occupied]) ** 2, axis=1) if np.any(occupied) else np.zeros(ham.shape[0])
    return density, eigvals


class MeanFieldSolver(BaseSolver):
    name = "mean_field"

    def __init__(self, settings: MeanFieldSettings | None = None) -> None:
        self.settings = settings or MeanFieldSettings()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family == "hubbard"

    def solve(self, problem: ProblemSpec) -> SolverResult:
        hopping = _hopping_matrix(problem)
        n_up, n_dn = _randomized_seed(problem, self.settings)
        converged = False
        iterations = 0
        final_delta = float("inf")

        for iterations in range(1, self.settings.max_iter + 1):
            h_up = hopping + np.diag(problem.U * n_dn - problem.mu)
            h_dn = hopping + np.diag(problem.U * n_up - problem.mu)
            new_up, eig_up = _occupy_negative_energy_states(h_up)
            new_dn, eig_dn = _occupy_negative_energy_states(h_dn)
            mixed_up = self.settings.mixing * new_up + (1.0 - self.settings.mixing) * n_up
            mixed_dn = self.settings.mixing * new_dn + (1.0 - self.settings.mixing) * n_dn
            delta = max(
                float(np.max(np.abs(mixed_up - n_up))),
                float(np.max(np.abs(mixed_dn - n_dn))),
            )
            final_delta = delta
            n_up, n_dn = mixed_up, mixed_dn
            if delta < self.settings.tol:
                converged = True
                break

        h_up = hopping + np.diag(problem.U * n_dn - problem.mu)
        h_dn = hopping + np.diag(problem.U * n_up - problem.mu)
        _, eig_up = _occupy_negative_energy_states(h_up)
        _, eig_dn = _occupy_negative_energy_states(h_dn)
        residual_up, _ = _occupy_negative_energy_states(h_up)
        residual_dn, _ = _occupy_negative_energy_states(h_dn)
        residual_norm = float(
            max(
                np.max(np.abs(residual_up - n_up)),
                np.max(np.abs(residual_dn - n_dn)),
            )
        )

        double_occ = np.clip(n_up * n_dn, 0.0, 1.0)
        sz_site = 0.5 * (n_up - n_dn)
        filling = float(np.sum(n_up + n_dn) / problem.nsites)
        d_avg = float(np.mean(double_occ))
        staggered = 0.0
        for y in range(problem.Ly):
            for x in range(problem.Lx):
                i = x + problem.Lx * y
                staggered += ((-1) ** (x + y)) * (n_up[i] - n_dn[i]) / problem.nsites
        ms2 = float(staggered**2)
        max_site = problem.nsites - 1
        cs_max = float(sz_site[0] * sz_site[max_site])

        rho_up = np.zeros_like(hopping)
        rho_dn = np.zeros_like(hopping)
        vals_up, vecs_up = np.linalg.eigh(h_up)
        vals_dn, vecs_dn = np.linalg.eigh(h_dn)
        for idx, energy in enumerate(vals_up):
            if energy < -1e-10:
                vec = vecs_up[:, idx]
                rho_up += np.outer(vec, vec.conj()).real
        for idx, energy in enumerate(vals_dn):
            if energy < -1e-10:
                vec = vecs_dn[:, idx]
                rho_dn += np.outer(vec, vec.conj()).real
        nbonds = len(nn_bonds(problem.Lx, problem.Ly))
        kinetic = float((np.trace(rho_up @ hopping) + np.trace(rho_dn @ hopping)).real / nbonds)
        energy = float(np.sum(vals_up[vals_up < -1e-10]) + np.sum(vals_dn[vals_dn < -1e-10]) - problem.U * np.sum(double_occ))

        bond_observables = {
            (i, j): float(sz_site[i] * sz_site[j])
            for i, j in nn_bonds(problem.Lx, problem.Ly)
        }

        return SolverResult(
            solver_name=self.name,
            energy=energy,
            global_observables={
                "D": d_avg,
                "n": filling,
                "Ms2": ms2,
                "K": kinetic,
                "Cs_max": cs_max,
                "energy": energy,
            },
            site_observables={
                "n_up": n_up.tolist(),
                "n_dn": n_dn.tolist(),
                "D_site": double_occ.tolist(),
                "Sz_site": sz_site.tolist(),
            },
            bond_observables=bond_observables,
            statevector=None,
            metadata={
                "converged": converged,
                "iterations": iterations,
                "final_delta": final_delta,
                "residual_norm": residual_norm,
                "seed": self.settings.seed,
                "init_noise_scale": self.settings.init_noise_scale,
            },
        )

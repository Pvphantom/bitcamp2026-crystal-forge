from __future__ import annotations

import math

import torch

from app.domain.problem_spec import ProblemSpec
from app.solvers.base import SolverResult


TRUST_FAMILY_FEATURE_DIM = 22


def build_trust_feature_vector(problem: ProblemSpec, cheap_result: SolverResult) -> torch.Tensor:
    if problem.model_family == "hubbard":
        return _build_hubbard_features(problem, cheap_result)
    if problem.model_family == "tfim":
        return _build_tfim_features(problem, cheap_result)
    raise ValueError(f"Unsupported model family for trust features: {problem.model_family}")


def _base_prefix(problem: ProblemSpec, param_a: float, param_b: float, param_c: float) -> list[float]:
    return [
        1.0 if problem.model_family == "hubbard" else 0.0,
        1.0 if problem.model_family == "tfim" else 0.0,
        float(problem.Lx),
        float(problem.Ly),
        float(problem.nsites),
        float(param_a),
        float(param_b),
        float(param_c),
    ]


def _stats(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std().item())


def _build_hubbard_features(problem: ProblemSpec, cheap_result: SolverResult) -> torch.Tensor:
    n_up = cheap_result.site_observables["n_up"]
    n_dn = cheap_result.site_observables["n_dn"]
    d_site = cheap_result.site_observables["D_site"]
    sz_site = cheap_result.site_observables["Sz_site"]
    abs_sz = [abs(value) for value in sz_site]
    density = [up + dn for up, dn in zip(n_up, n_dn, strict=True)]
    staggered_linear = 0.0
    for idx, sz in enumerate(sz_site):
        x = idx % problem.Lx
        y = idx // problem.Lx
        staggered_linear += ((-1) ** (x + y)) * (n_up[idx] - n_dn[idx]) / problem.nsites
    return torch.tensor(
        _base_prefix(problem, problem.t, problem.U, problem.mu)
        + [
            cheap_result.global_observables["D"],
            cheap_result.global_observables["n"],
            cheap_result.global_observables["Ms2"],
            cheap_result.global_observables["K"],
            cheap_result.global_observables["Cs_max"],
            cheap_result.global_observables["energy"],
            float(sum(abs_sz) / len(abs_sz)),
            float(max(abs_sz)),
            float(staggered_linear),
            float(torch.tensor(density).std().item()),
            float(torch.tensor(sz_site).std().item()),
            float(torch.tensor(d_site).std().item()),
            1.0 if cheap_result.metadata.get("converged", False) else 0.0,
            float(cheap_result.metadata.get("iterations", 0)),
        ],
        dtype=torch.float32,
    )


def _build_tfim_features(problem: ProblemSpec, cheap_result: SolverResult) -> torch.Tensor:
    mz = cheap_result.site_observables["Mz_site"]
    mx = cheap_result.site_observables["Mx_site"]
    abs_mz = [abs(value) for value in mz]
    polar = [math.sqrt(x * x + z * z) for x, z in zip(mx, mz, strict=True)]
    staggered_linear = 0.0
    for idx, value in enumerate(mz):
        x = idx % problem.Lx
        y = idx // problem.Lx
        staggered_linear += ((-1) ** (x + y)) * value / problem.nsites
    return torch.tensor(
        _base_prefix(problem, problem.J, problem.h, problem.g)
        + [
            cheap_result.global_observables["Mz"],
            cheap_result.global_observables["Mx"],
            cheap_result.global_observables["ZZ_nn"],
            cheap_result.global_observables["Mstag2"],
            cheap_result.global_observables["Z_span"],
            cheap_result.global_observables["energy"],
            float(sum(abs_mz) / len(abs_mz)),
            float(max(abs_mz)),
            float(staggered_linear),
            float(torch.tensor(mx).std().item()),
            float(torch.tensor(mz).std().item()),
            float(torch.tensor(polar).std().item()),
            1.0 if cheap_result.metadata.get("converged", False) else 0.0,
            float(cheap_result.metadata.get("iterations", 0)),
        ],
        dtype=torch.float32,
    )

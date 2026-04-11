from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.analysis.solver_compare import compare_solver_results
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR, DEFAULT_TRUST_DATASET
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver


GRIDS = {
    (2, 2): {
        "U": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        "mu": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    (2, 3): {
        "U": [0.0, 2.0, 4.0, 8.0, 10.0],
        "mu": [0.0, 2.0, 4.0, 5.0],
    },
}


def build_trust_feature_vector(
    *,
    problem: ProblemSpec,
    mean_field_globals: dict[str, float],
    mean_field_sites: dict[str, list[float]],
    iterations: int,
    converged: bool,
) -> torch.Tensor:
    n_up = mean_field_sites["n_up"]
    n_dn = mean_field_sites["n_dn"]
    d_site = mean_field_sites["D_site"]
    sz_site = mean_field_sites["Sz_site"]
    abs_sz = [abs(value) for value in sz_site]
    density = [up + dn for up, dn in zip(n_up, n_dn, strict=True)]
    staggered_linear = 0.0
    for idx, sz in enumerate(sz_site):
        x = idx % problem.Lx
        y = idx // problem.Lx
        staggered_linear += ((-1) ** (x + y)) * (n_up[idx] - n_dn[idx]) / problem.nsites
    return torch.tensor(
        [
            float(problem.Lx),
            float(problem.Ly),
            float(problem.nsites),
            problem.t,
            problem.U,
            problem.mu,
            mean_field_globals["D"],
            mean_field_globals["n"],
            mean_field_globals["Ms2"],
            mean_field_globals["K"],
            mean_field_globals["Cs_max"],
            mean_field_globals["energy"],
            float(sum(abs_sz) / len(abs_sz)),
            float(max(abs_sz)),
            float(staggered_linear),
            float(torch.tensor(density).std().item()),
            float(torch.tensor(sz_site).std().item()),
            float(torch.tensor(d_site).std().item()),
            1.0 if converged else 0.0,
            float(iterations),
        ],
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_TRUST_DATASET)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    exact_solver = ExactEDSolver()
    mean_field_solver = MeanFieldSolver()

    dataset = []
    sample_id = 0
    for (Lx, Ly), grid in GRIDS.items():
        for U in grid["U"]:
            for mu in grid["mu"]:
                problem = ProblemSpec.hubbard(Lx=Lx, Ly=Ly, t=1.0, U=U, mu=mu)
                exact = exact_solver.solve(problem)
                approx = mean_field_solver.solve(problem)
                comparison = compare_solver_results(problem, exact, approx)
                sample_id += 1
                dataset.append(
                    {
                        "features": build_trust_feature_vector(
                            problem=problem,
                            mean_field_globals=approx.global_observables,
                            mean_field_sites=approx.site_observables,
                            iterations=int(approx.metadata["iterations"]),
                            converged=bool(approx.metadata["converged"]),
                        ),
                        "risk_label": comparison.risk_label,
                        "max_abs_error": comparison.max_abs_error,
                        "energy_error": comparison.energy_error,
                        "metadata": {
                            "id": sample_id,
                            "Lx": Lx,
                            "Ly": Ly,
                            "nsites": problem.nsites,
                            "t": problem.t,
                            "U": problem.U,
                            "mu": problem.mu,
                        },
                    }
                )

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} TrustNet samples to {args.output}")


if __name__ == "__main__":
    main()

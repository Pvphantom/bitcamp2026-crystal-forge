from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.analysis.solver_compare import compare_solver_results
from app.analysis.trust_features import build_trust_feature_vector
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR, DEFAULT_TRUST_DATASET
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


GRIDS = {
    ("hubbard", 2, 2): {
        "U": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        "mu": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    ("hubbard", 2, 3): {
        "U": [0.0, 2.0, 4.0, 8.0, 10.0],
        "mu": [0.0, 2.0, 4.0, 5.0],
    },
    ("tfim", 2, 2): {
        "J": [0.5, 1.0, 1.5],
        "h": [0.2, 0.6, 1.0, 1.5, 2.0],
        "g": [0.0, 0.5],
    },
    ("tfim", 2, 3): {
        "J": [1.0],
        "h": [0.4, 1.0, 1.8],
        "g": [0.0, 0.5],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_TRUST_DATASET)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    exact_solver = ExactEDSolver()
    mean_field_solver = MeanFieldSolver()
    tfim_mean_field_solver = TFIMMeanFieldSolver()

    dataset = []
    sample_id = 0
    for (family, Lx, Ly), grid in GRIDS.items():
        if family == "hubbard":
            for U in grid["U"]:
                for mu in grid["mu"]:
                    problem = ProblemSpec.hubbard(Lx=Lx, Ly=Ly, t=1.0, U=U, mu=mu)
                    exact = exact_solver.solve(problem)
                    approx = mean_field_solver.solve(problem)
                    comparison = compare_solver_results(problem, exact, approx)
                    sample_id += 1
                    dataset.append(
                        {
                            "features": build_trust_feature_vector(problem, approx),
                            "risk_label": comparison.risk_label,
                            "max_abs_error": comparison.max_abs_error,
                            "energy_error": comparison.energy_error,
                            "metadata": {
                                "id": sample_id,
                                "family": family,
                                "Lx": Lx,
                                "Ly": Ly,
                                "nsites": problem.nsites,
                                "t": problem.t,
                                "U": problem.U,
                                "mu": problem.mu,
                            },
                        }
                    )
        elif family == "tfim":
            for J in grid["J"]:
                for h in grid["h"]:
                    for g in grid["g"]:
                        problem = ProblemSpec.tfim(Lx=Lx, Ly=Ly, J=J, h=h, g=g)
                        exact = exact_solver.solve(problem)
                        approx = tfim_mean_field_solver.solve(problem)
                        comparison = compare_solver_results(problem, exact, approx)
                        sample_id += 1
                        dataset.append(
                            {
                                "features": build_trust_feature_vector(problem, approx),
                                "risk_label": comparison.risk_label,
                                "max_abs_error": comparison.max_abs_error,
                                "energy_error": comparison.energy_error,
                                "metadata": {
                                    "id": sample_id,
                                    "family": family,
                                    "Lx": Lx,
                                    "Ly": Ly,
                                    "nsites": problem.nsites,
                                    "J": problem.J,
                                    "h": problem.h,
                                    "g": problem.g,
                                },
                            }
                        )

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} TrustNet samples to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.ml.schema import ARTIFACTS_DIR, DEFAULT_QPROBE_DATASET
from app.optimization.measurement_plan import search_minimal_measurement_plan
from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.observables import (
    build_double_occ,
    build_filling,
    build_kinetic,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
)


TARGET_SETS = [
    ("D", "n", "Ms2", "Cs_max"),
    ("D", "n", "Ms2", "K", "Cs_max"),
    ("K",),
]
NOISE_LEVELS = [0.02, 0.08]
SHOT_COUNTS = [2000, 4000]
TOLERANCES = [0.01, 0.03]
PARAMETER_POINTS = [
    {"U": 0.5, "mu": 0.0},
    {"U": 4.0, "mu": 2.0},
    {"U": 8.0, "mu": 4.0},
    {"U": 8.0, "mu": 1.0},
]


def build_feature_vector(*, U: float, mu: float, D: float, n: float, Ms2: float, K: float, Cs_max: float,
                         tolerance: float, shots_per_group: int, readout_flip_prob: float,
                         targets: tuple[str, ...]) -> torch.Tensor:
    target_flags = [1.0 if name in targets else 0.0 for name in ["D", "n", "Ms2", "K", "Cs_max"]]
    return torch.tensor(
        [
            U,
            mu,
            D,
            n,
            Ms2,
            K,
            Cs_max,
            tolerance,
            float(shots_per_group),
            readout_flip_prob,
            *target_flags,
        ],
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_QPROBE_DATASET)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = []
    sample_id = 0

    for point in PARAMETER_POINTS:
        U = point["U"]
        mu = point["mu"]
        h_op = build_hamiltonian(2, 2, t=1.0, U=U, mu=mu)
        _, state = ground_state(h_op)

        D = expectation_value(build_double_occ(2, 2), state)
        n = expectation_value(build_filling(2, 2), state)
        Ms2 = expectation_value(build_staggered_magnetization_squared(2, 2), state)
        K = expectation_value(build_kinetic(2, 2, t=1.0), state)
        Cs_max = expectation_value(build_spin_correlator_maxdist(2, 2), state)

        for targets in TARGET_SETS:
            for readout_flip_prob in NOISE_LEVELS:
                for shots_per_group in SHOT_COUNTS:
                    for tolerance in TOLERANCES:
                        result = search_minimal_measurement_plan(
                            Lx=2,
                            Ly=2,
                            t=1.0,
                            state=state,
                            target_observables=targets,
                            tolerance=tolerance,
                            shots_per_group=shots_per_group,
                            noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                            seed=11,
                        )
                        sample_id += 1
                        dataset.append(
                            {
                                "features": build_feature_vector(
                                    U=U,
                                    mu=mu,
                                    D=D,
                                    n=n,
                                    Ms2=Ms2,
                                    K=K,
                                    Cs_max=Cs_max,
                                    tolerance=tolerance,
                                    shots_per_group=shots_per_group,
                                    readout_flip_prob=readout_flip_prob,
                                    targets=targets,
                                ),
                                "recommended_cost": result.recommended_plan.cost,
                                "full_cost": result.full_plan.cost,
                                "measurement_savings": result.full_plan.cost - result.recommended_plan.cost,
                                "success": result.success,
                                "max_abs_error": result.max_abs_error,
                                "group_bases": result.recommended_plan.bases,
                                "targets": list(targets),
                                "metadata": {
                                    "id": sample_id,
                                    "U": U,
                                    "mu": mu,
                                    "targets": list(targets),
                                    "tolerance": tolerance,
                                    "shots_per_group": shots_per_group,
                                    "readout_flip_prob": readout_flip_prob,
                                },
                            }
                        )

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} ML-QProbe samples to {args.output}")


if __name__ == "__main__":
    main()

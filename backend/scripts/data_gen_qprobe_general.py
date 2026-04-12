from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_operator_features import build_qprobe_operator_feature_vector
from app.ml.schema import ARTIFACTS_DIR, DEFAULT_QPROBE_GENERAL_DATASET
from app.observables.registry import build_default_observable_registry
from app.optimization.measurement_plan import search_minimal_measurement_plan_for_problem
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.tfim import build_tfim_hamiltonian


NOISE_LEVELS = [0.02, 0.08]
SHOT_COUNTS = [2000, 4000]
TOLERANCES = [0.01, 0.03]

HUBBARD_POINTS = [
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 0.5, "mu": 0.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 4.0},
]

TFIM_POINTS = [
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 0.8, "g": 0.0},
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 1.8, "g": 0.5},
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 2.4, "g": 0.0},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_QPROBE_GENERAL_DATASET)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    registry = build_default_observable_registry()
    dataset: list[dict] = []
    sample_id = 0

    for point in HUBBARD_POINTS:
        problem = ProblemSpec.hubbard(**point)
        dataset.extend(_samples_for_problem(problem, registry, sample_id_start=sample_id + 1))
        sample_id = len(dataset)
    for point in TFIM_POINTS:
        problem = ProblemSpec.tfim(**point)
        dataset.extend(_samples_for_problem(problem, registry, sample_id_start=sample_id + 1))
        sample_id = len(dataset)

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} general ML-QProbe samples to {args.output}")


def _samples_for_problem(problem: ProblemSpec, registry, *, sample_id_start: int) -> list[dict]:
    operator_map = registry.operator_map(problem)
    names = registry.names_for_family(problem.model_family)
    targets_list = []
    for r in range(1, min(3, len(names)) + 1):
        targets_list.extend(combinations(names, r))
    state = _ground_state(problem)
    rows: list[dict] = []
    sample_id = sample_id_start
    for targets in targets_list:
        for readout_flip_prob in NOISE_LEVELS:
            for shots_per_group in SHOT_COUNTS:
                for tolerance in TOLERANCES:
                    result = search_minimal_measurement_plan_for_problem(
                        problem=problem,
                        state=state,
                        target_observables=tuple(targets),
                        tolerance=tolerance,
                        shots_per_group=shots_per_group,
                        noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                        seed=11,
                        registry=registry,
                    )
                    rows.append(
                        {
                            "features": build_qprobe_operator_feature_vector(
                                problem=problem,
                                operator_map=operator_map,
                                target_names=tuple(targets),
                                tolerance=tolerance,
                                shots_per_group=shots_per_group,
                                readout_flip_prob=readout_flip_prob,
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
                                "family": problem.model_family,
                                "Lx": problem.Lx,
                                "Ly": problem.Ly,
                                "parameters": dict(problem.parameters.values),
                                "targets": list(targets),
                                "tolerance": tolerance,
                                "shots_per_group": shots_per_group,
                                "readout_flip_prob": readout_flip_prob,
                            },
                        }
                    )
                    sample_id += 1
    return rows


def _ground_state(problem: ProblemSpec):
    if problem.model_family == "hubbard":
        h_op = build_hamiltonian(problem.Lx, problem.Ly, problem.t, problem.U, problem.mu)
    else:
        h_op = build_tfim_hamiltonian(problem.Lx, problem.Ly, problem.J, problem.h, problem.g)
    _, state = ground_state(h_op)
    return state


if __name__ == "__main__":
    main()

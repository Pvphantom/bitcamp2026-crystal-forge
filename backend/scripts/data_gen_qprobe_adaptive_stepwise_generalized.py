from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.analysis.synthetic_operator_families_generalized import (
    GENERALIZED_OPERATOR_FAMILIES,
    HARD_OPERATOR_FAMILIES,
    build_generalized_synthetic_operator_bundle,
)
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_adaptive_step_features import build_qprobe_adaptive_step_feature_vector
from app.ml.schema import ARTIFACTS_DIR
from app.optimization.measurement_plan import search_adaptive_measurement_plan_with_operator_map
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.tfim import build_tfim_hamiltonian


NOISE_LEVELS = [0.02, 0.08]
SHOT_COUNTS = [2000, 4000]
TOLERANCES = [0.01, 0.02, 0.03]
NUM_TARGET_OPTIONS = [1, 2, 3]
BASE_SEEDS = [5, 11, 19]
HARD_EXTRA_SEEDS = [23, 29, 31, 37]

HUBBARD_POINTS = [
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 0.5, "mu": 0.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0},
]

TFIM_POINTS = [
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 0.8, "g": 0.0},
    {"Lx": 2, "Ly": 2, "J": 1.0, "h": 1.8, "g": 0.5},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "qprobe_adaptive_stepwise_generalized_dataset.pt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset: list[dict] = []
    sample_id = 0

    for point in HUBBARD_POINTS:
        problem = ProblemSpec.hubbard(**point)
        dataset.extend(_samples_for_problem(problem, sample_id_start=sample_id + 1, quick=args.quick))
        sample_id = len(dataset)
    for point in TFIM_POINTS:
        problem = ProblemSpec.tfim(**point)
        dataset.extend(_samples_for_problem(problem, sample_id_start=sample_id + 1, quick=args.quick))
        sample_id = len(dataset)

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} generalized adaptive stepwise ML-QProbe samples to {args.output}")


def _samples_for_problem(problem: ProblemSpec, *, sample_id_start: int, quick: bool) -> list[dict]:
    state = _ground_state(problem)
    rows: list[dict] = []
    sample_id = sample_id_start
    noise_levels = [0.02] if quick else NOISE_LEVELS
    shot_counts = [2000] if quick else SHOT_COUNTS
    tolerances = [0.02, 0.03] if quick else TOLERANCES
    num_target_options = [1, 2] if quick else NUM_TARGET_OPTIONS
    for family in GENERALIZED_OPERATOR_FAMILIES:
        seeds = ([5, 11] if quick else BASE_SEEDS) + (([23, 29] if quick else HARD_EXTRA_SEEDS) if family in HARD_OPERATOR_FAMILIES else [])
        for num_targets in num_target_options:
            for synth_seed in seeds:
                bundle = build_generalized_synthetic_operator_bundle(
                    problem=problem,
                    family=family,
                    num_targets=num_targets,
                    seed=synth_seed,
                )
                for readout_flip_prob in noise_levels:
                    for shots_per_group in shot_counts:
                        for tolerance in tolerances:
                            result = search_adaptive_measurement_plan_with_operator_map(
                                state=state,
                                operator_map=bundle.operator_map,
                                target_observables=bundle.target_names,
                                tolerance=tolerance,
                                shots_per_group=shots_per_group,
                                noise_model=NoiseModel(readout_flip_prob=readout_flip_prob),
                                seed=17,
                            )
                            for step in result.steps:
                                margin = float(tolerance - step.max_abs_error)
                                safe_stop = bool(not step.unresolved_targets and margin >= 0.0)
                                coverage_complete = bool(not step.unresolved_targets)
                                rows.append(
                                    {
                                        "features": build_qprobe_adaptive_step_feature_vector(
                                            problem=problem,
                                            operator_map=bundle.operator_map,
                                            target_names=bundle.target_names,
                                            tolerance=tolerance,
                                            shots_per_group=shots_per_group,
                                            readout_flip_prob=readout_flip_prob,
                                            step=step,
                                            full_cost=result.full_plan.cost,
                                        ),
                                        "safe_stop": safe_stop,
                                        "coverage_complete": coverage_complete,
                                        "margin": margin,
                                        "current_cost": step.plan.cost,
                                        "final_cost": result.final_plan.cost,
                                        "full_cost": result.full_plan.cost,
                                        "metadata": {
                                            "id": sample_id,
                                            "family": problem.model_family,
                                            "operator_family": family,
                                            "num_targets": num_targets,
                                            "Lx": problem.Lx,
                                            "Ly": problem.Ly,
                                            "parameters": dict(problem.parameters.values),
                                            "targets": list(bundle.target_names),
                                            "tolerance": tolerance,
                                            "shots_per_group": shots_per_group,
                                            "readout_flip_prob": readout_flip_prob,
                                            "synth_seed": synth_seed,
                                            "step_index": step.step_index,
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

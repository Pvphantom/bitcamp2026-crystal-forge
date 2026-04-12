from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_superconductor_features import build_qprobe_superconductor_feature_vector
from app.observables.registry import build_default_observable_registry
from app.optimization.adaptive_bounded import search_bounded_adaptive_plan_with_operator_map
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel


POINTS = [
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 2.0, "mu": 0.5},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 1.0},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 6.0, "mu": 1.5},
    {"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 2.0},
]

BUNDLES: dict[str, tuple[str, ...]] = {
    "charge_sector": ("n", "D"),
    "spin_sector": ("Ms2", "Cs_max"),
    "pair_sector": ("Pair_nn", "Pair_span"),
    "transport": ("K",),
    "transport_pairing": ("K", "Pair_nn", "Pair_span"),
    "competing_orders": ("n", "Ms2", "Pair_nn", "Pair_span"),
    "superconductor_panel": ("D", "K", "Ms2", "Pair_nn", "Pair_span"),
}

TOLERANCES = [0.03, 0.05, 0.08, 0.10, 0.15]
SHOT_COUNTS = [2000, 4000]
NOISE_LEVELS = [0.02, 0.08]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("backend/artifacts/qprobe_superconductor_ml_dataset.pt"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--node-budget", type=int, default=64)
    args = parser.parse_args()

    samples = build_dataset(quick=args.quick, node_budget=args.node_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out)
    print(f"saved {len(samples)} superconductivity ML-QProbe samples to {args.out}")


def build_dataset(*, quick: bool, node_budget: int) -> list[dict]:
    registry = build_default_observable_registry()
    points = POINTS[:3] if quick else POINTS
    bundles = {k: v for k, v in BUNDLES.items() if k in {"charge_sector", "spin_sector", "pair_sector", "transport", "transport_pairing"}} if quick else BUNDLES
    tolerances = TOLERANCES[:3] if quick else TOLERANCES
    shots = SHOT_COUNTS[:1] if quick else SHOT_COUNTS
    noises = NOISE_LEVELS[:1] if quick else NOISE_LEVELS

    samples: list[dict] = []
    for point in points:
        problem = ProblemSpec.hubbard(**point)
        operator_bank = {name: registry.operator(name, problem) for name in registry.names_for_family("hubbard")}
        _, state = ground_state(build_hamiltonian(problem.Lx, problem.Ly, problem.t, problem.U, problem.mu))
        for bundle_name, targets in bundles.items():
            submap = {name: operator_bank[name] for name in targets}
            for tolerance in tolerances:
                for shots_per_group in shots:
                    for readout_flip_prob in noises:
                        noise = NoiseModel(readout_flip_prob=readout_flip_prob)
                        bounded, meta = search_bounded_adaptive_plan_with_operator_map(
                            state=state,
                            operator_map=submap,
                            target_observables=targets,
                            tolerance=tolerance,
                            shots_per_group=shots_per_group,
                            noise_model=noise,
                            seed=17,
                            node_budget=node_budget,
                        )
                        features = build_qprobe_superconductor_feature_vector(
                            problem=problem,
                            operator_map=submap,
                            target_names=targets,
                            tolerance=tolerance,
                            shots_per_group=shots_per_group,
                            readout_flip_prob=readout_flip_prob,
                        )
                        samples.append(
                            {
                                "features": features,
                                "safe": bool(bounded.oracle_benchmark_within_tolerance),
                                "metadata": {
                                    "bundle_name": bundle_name,
                                    "problem_key": f"U={problem.U}|mu={problem.mu}",
                                    "tolerance": tolerance,
                                    "shots_per_group": shots_per_group,
                                    "readout_flip_prob": readout_flip_prob,
                                    "bounded_cost": int(bounded.final_plan.cost),
                                    "lower_bound": int(meta.certified_lower_bound),
                                },
                            }
                        )
    return samples


if __name__ == "__main__":
    main()

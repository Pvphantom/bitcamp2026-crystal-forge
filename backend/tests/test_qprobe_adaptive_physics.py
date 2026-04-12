from __future__ import annotations

import torch

from app.analysis.synthetic_operator_families import build_synthetic_operator_bundle
from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_adaptive_physics_features import (
    build_qprobe_adaptive_physics_feature_vector,
    qprobe_adaptive_physics_feature_dim,
)
from app.optimization.measurement_plan import search_adaptive_measurement_plan_with_operator_map
from app.physics.ed import ground_state
from app.physics.measurement_eval import NoiseModel
from app.physics.tfim import build_tfim_hamiltonian
from scripts.train_qprobe_adaptive_physics_model import split_by_operator_family


def test_physics_feature_vector_has_expected_dimension() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=1.0, g=0.0)
    bundle = build_synthetic_operator_bundle(problem=problem, family="pair_zz", num_targets=2, seed=7)
    _, state = ground_state(build_tfim_hamiltonian(problem.Lx, problem.Ly, problem.J, problem.h, problem.g))
    result = search_adaptive_measurement_plan_with_operator_map(
        state=state,
        operator_map=bundle.operator_map,
        target_observables=bundle.target_names,
        tolerance=0.03,
        shots_per_group=2000,
        noise_model=NoiseModel(readout_flip_prob=0.02),
        seed=17,
    )
    features = build_qprobe_adaptive_physics_feature_vector(
        problem=problem,
        operator_map=bundle.operator_map,
        target_names=bundle.target_names,
        tolerance=0.03,
        shots_per_group=2000,
        readout_flip_prob=0.02,
        step=result.steps[0],
        full_cost=result.full_plan.cost,
    )
    assert features.shape == torch.Size([qprobe_adaptive_physics_feature_dim()])


def test_physics_split_prevents_operator_family_leakage() -> None:
    samples = []
    for family in ("hubbard", "tfim"):
        for op_family in ("local_z", "mixed_local", "pair_zz", "pair_xx"):
            for _ in range(2):
                samples.append(
                    {
                        "features": torch.zeros(qprobe_adaptive_physics_feature_dim()),
                        "coverage_complete": False,
                        "safe_stop": False,
                        "margin": -0.01,
                        "metadata": {"family": family, "operator_family": op_family},
                    }
                )
    train, val, test = split_by_operator_family(samples)
    tg = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in train}
    vg = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in val}
    sg = {f"{s['metadata']['family']}|{s['metadata']['operator_family']}" for s in test}
    assert tg.isdisjoint(vg)
    assert tg.isdisjoint(sg)
    assert vg.isdisjoint(sg)

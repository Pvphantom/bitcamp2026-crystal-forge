"""Dataset generation helpers for mixed-size lattice data."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sklearn.model_selection import train_test_split

from app.ml.schema import build_graph_sample, classify_phase_rule
from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.lattice import nn_bonds
from app.physics.observables import (
    build_bond_spin_correlator_operators,
    build_double_occ,
    build_filling,
    build_kinetic,
    build_staggered_magnetization_squared,
    extract_site_observables_from_statevector,
)


U_GRID = [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10]
MU_GRID = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
LARGE_LATTICE_POINTS = [
    (0.5, 0.0),   # Metal-like
    (3.0, 0.5),   # Singlet-rich-like
    (5.0, 2.0),   # Antiferromagnetic-like
    (8.0, 2.0),   # Mott-like
]


def parameter_grid(*, large_lattice: bool = False) -> Iterable[tuple[float, float]]:
    if large_lattice:
        for point in LARGE_LATTICE_POINTS:
            yield point
        return

    u_values = U_GRID
    mu_values = MU_GRID
    for U in u_values:
        for mu in mu_values:
            yield U, mu


def generate_base_samples(
    *,
    Lx: int,
    Ly: int,
    max_nodes: int,
    grid: Iterable[tuple[float, float]] | None = None,
) -> list[dict]:
    samples: list[dict] = []
    grid = grid if grid is not None else parameter_grid(large_lattice=(Lx * Ly > 4))

    for U, mu in grid:
        h_op = build_hamiltonian(Lx, Ly, t=1.0, U=U, mu=mu)
        _, psi0 = ground_state(h_op)
        site_obs = extract_site_observables_from_statevector(Lx, Ly, psi0)
        bond_ops = build_bond_spin_correlator_operators(Lx, Ly)
        bond_strengths = {
            bond: expectation_value(op, psi0)
            for bond, op in bond_ops.items()
        }
        d_val = expectation_value(build_double_occ(Lx, Ly), psi0)
        n_val = expectation_value(build_filling(Lx, Ly), psi0)
        ms2_val = expectation_value(build_staggered_magnetization_squared(Lx, Ly), psi0)
        k_val = expectation_value(build_kinetic(Lx, Ly, t=1.0), psi0)

        node_features = []
        for y in range(Ly):
            for x in range(Lx):
                i = x + Lx * y
                node_features.append(
                    [
                        site_obs["n_up"][i],
                        site_obs["n_dn"][i],
                        site_obs["D_site"][i],
                        site_obs["Sz_site"][i],
                        1.0 if (x + y) % 2 == 0 else -1.0,
                    ]
                )

        label = classify_phase_rule(U, n_val, ms2_val)
        sample = build_graph_sample(
            Lx=Lx,
            Ly=Ly,
            site_features=node_features,
            bond_strengths=bond_strengths,
            global_feats=[U, mu, float(Lx * Ly)],
            label=label,
            metadata={
                "base_id": f"{Lx}x{Ly}:U={U}:mu={mu}",
                "Lx": Lx,
                "Ly": Ly,
                "U": U,
                "mu": mu,
                "label": label,
                "D": d_val,
                "n": n_val,
                "Ms2": ms2_val,
                "K": k_val,
                "bonds": nn_bonds(Lx, Ly),
            },
            max_nodes=max_nodes,
        )
        samples.append(sample.to_dict())
    return samples


def augment_samples(samples: list[dict], *, copies: int = 4, sigma: float = 0.015) -> list[dict]:
    augmented = list(samples)
    for sample in samples:
        for copy_idx in range(copies):
            noisy = {**sample}
            noisy["nodes"] = sample["nodes"] + sigma * torch.randn_like(sample["nodes"])
            noisy["edge_attr"] = sample["edge_attr"] + sigma * torch.randn_like(sample["edge_attr"])
            noisy["metadata"] = {**sample["metadata"], "augmentation": copy_idx + 1}
            augmented.append(noisy)
    return augmented


def split_base_samples(base_samples: list[dict], *, test_size: float = 0.2) -> tuple[list[dict], list[dict]]:
    labels = [sample["label"] for sample in base_samples]
    return train_test_split(
        base_samples,
        test_size=test_size,
        random_state=7,
        stratify=labels,
    )

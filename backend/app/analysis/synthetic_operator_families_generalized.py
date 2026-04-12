from __future__ import annotations

from dataclasses import dataclass
import random

from qiskit.quantum_info import SparsePauliOp

from app.analysis.synthetic_operator_families import (
    _distinct_pair,
    _multi_site_pauli,
    _rand_coeff,
    _single_site_pauli,
    _two_site_pauli,
)
from app.domain.problem_spec import ProblemSpec


GENERALIZED_OPERATOR_FAMILIES = (
    "local_z",
    "local_x",
    "pair_zz",
    "pair_xx",
    "mixed_local",
    "long_range_mixed",
    "pair_xy",
    "cluster_mixed",
    "diffuse_diagonal",
    "diffuse_mixed",
    "basis_conflict",
)

HARD_OPERATOR_FAMILIES = {
    "mixed_local",
    "long_range_mixed",
    "pair_xy",
    "cluster_mixed",
    "diffuse_mixed",
    "basis_conflict",
}


@dataclass(frozen=True)
class SyntheticOperatorBundle:
    family: str
    operator_map: dict[str, SparsePauliOp]
    target_names: tuple[str, ...]


def build_generalized_synthetic_operator_bundle(
    *,
    problem: ProblemSpec,
    family: str,
    num_targets: int,
    seed: int,
) -> SyntheticOperatorBundle:
    if family not in GENERALIZED_OPERATOR_FAMILIES:
        raise ValueError(f"unsupported generalized synthetic operator family: {family}")
    rng = random.Random(seed)
    nqubits = problem.nqubits if problem.model_family == "hubbard" else problem.nsites
    operator_map: dict[str, SparsePauliOp] = {}
    for index in range(num_targets):
        name = f"{family}_{index + 1}"
        operator_map[name] = _build_operator(problem=problem, family=family, nqubits=nqubits, rng=rng)
    return SyntheticOperatorBundle(family=family, operator_map=operator_map, target_names=tuple(operator_map.keys()))


def _build_operator(*, problem: ProblemSpec, family: str, nqubits: int, rng: random.Random) -> SparsePauliOp:
    if family == "local_z":
        terms = [(_single_site_pauli(nqubits, rng.randrange(nqubits), "Z"), _rand_coeff(rng))]
        if rng.random() < 0.5:
            terms.append((_single_site_pauli(nqubits, rng.randrange(nqubits), "Z"), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "local_x":
        terms = [(_single_site_pauli(nqubits, rng.randrange(nqubits), "X"), _rand_coeff(rng))]
        if rng.random() < 0.5:
            terms.append((_single_site_pauli(nqubits, rng.randrange(nqubits), "X"), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "pair_zz":
        i, j = _distinct_pair(nqubits, rng)
        terms = [(_two_site_pauli(nqubits, i, j, "Z", "Z"), _rand_coeff(rng))]
        if rng.random() < 0.5:
            i2, j2 = _distinct_pair(nqubits, rng)
            terms.append((_two_site_pauli(nqubits, i2, j2, "Z", "Z"), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "pair_xx":
        i, j = _distinct_pair(nqubits, rng)
        terms = [(_two_site_pauli(nqubits, i, j, "X", "X"), _rand_coeff(rng))]
        if rng.random() < 0.5:
            i2, j2 = _distinct_pair(nqubits, rng)
            terms.append((_two_site_pauli(nqubits, i2, j2, "X", "X"), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "mixed_local":
        i = rng.randrange(nqubits)
        j = rng.randrange(nqubits)
        terms = [
            (_single_site_pauli(nqubits, i, "Z"), _rand_coeff(rng)),
            (_single_site_pauli(nqubits, j, "X"), _rand_coeff(rng)),
        ]
        if rng.random() < 0.5:
            k, l = _distinct_pair(nqubits, rng)
            terms.append((_two_site_pauli(nqubits, k, l, "Z", "X"), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "pair_xy":
        i, j = _distinct_pair(nqubits, rng)
        terms = [(_two_site_pauli(nqubits, i, j, rng.choice(["X", "Y"]), rng.choice(["X", "Y", "Z"])), _rand_coeff(rng))]
        if rng.random() < 0.7:
            i2, j2 = _distinct_pair(nqubits, rng)
            terms.append((_two_site_pauli(nqubits, i2, j2, rng.choice(["X", "Y", "Z"]), rng.choice(["X", "Y"])), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "cluster_mixed":
        support = min(max(3, nqubits // 3 + 1), nqubits)
        terms: list[tuple[str, float]] = []
        for _ in range(2 + (1 if rng.random() < 0.7 else 0)):
            indices = sorted(rng.sample(list(range(nqubits)), k=support))
            symbols = [rng.choice(["X", "Y", "Z"]) for _ in indices]
            terms.append((_multi_site_pauli(nqubits, indices, symbols), _rand_coeff(rng)))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "diffuse_diagonal":
        count = min(max(3, nqubits // 2), nqubits)
        terms = [(_single_site_pauli(nqubits, idx, "Z"), 0.25) for idx in rng.sample(list(range(nqubits)), k=count)]
        if count >= 2:
            i, j = _distinct_pair(nqubits, rng)
            terms.append((_two_site_pauli(nqubits, i, j, "Z", "Z"), 0.25))
        return SparsePauliOp.from_list(terms).simplify()
    if family == "basis_conflict":
        support = min(max(2, nqubits // 3 + 1), nqubits)
        indices = sorted(rng.sample(list(range(nqubits)), k=support))
        symbol_sets = [
            [rng.choice(["X", "Y", "Z"]) for _ in indices],
            [rng.choice(["X", "Y", "Z"]) for _ in indices],
            [rng.choice(["X", "Y", "Z"]) for _ in indices],
        ]
        terms = [(_multi_site_pauli(nqubits, indices, symbols), _rand_coeff(rng)) for symbols in symbol_sets]
        return SparsePauliOp.from_list(terms).simplify()
    # long_range_mixed and diffuse_mixed both live here.
    term_count = 2 + (1 if rng.random() < 0.8 else 0) + (1 if family == "diffuse_mixed" and rng.random() < 0.6 else 0)
    terms: list[tuple[str, float]] = []
    for _ in range(term_count):
        if family == "long_range_mixed":
            support = min(max(3, nqubits // 2), nqubits)
        else:
            support = min(max(2, nqubits // 3 + rng.choice([0, 1, 2])), nqubits)
        indices = sorted(rng.sample(list(range(nqubits)), k=support))
        symbols = [rng.choice(["X", "Y", "Z"]) for _ in indices]
        coeff = 0.25 if family == "diffuse_mixed" else _rand_coeff(rng)
        terms.append((_multi_site_pauli(nqubits, indices, symbols), coeff))
    return SparsePauliOp.from_list(terms).simplify()

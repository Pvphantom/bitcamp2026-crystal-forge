from __future__ import annotations

from dataclasses import dataclass
import random

from qiskit.quantum_info import SparsePauliOp

from app.domain.problem_spec import ProblemSpec


SYNTHETIC_OPERATOR_FAMILIES = (
    "local_z",
    "local_x",
    "pair_zz",
    "pair_xx",
    "mixed_local",
    "long_range_mixed",
)


@dataclass(frozen=True)
class SyntheticOperatorBundle:
    family: str
    operator_map: dict[str, SparsePauliOp]
    target_names: tuple[str, ...]


def build_synthetic_operator_bundle(
    *,
    problem: ProblemSpec,
    family: str,
    num_targets: int,
    seed: int,
) -> SyntheticOperatorBundle:
    if family not in SYNTHETIC_OPERATOR_FAMILIES:
        raise ValueError(f"unsupported synthetic operator family: {family}")
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
    support = min(max(3, nqubits // 2), nqubits)
    indices = rng.sample(list(range(nqubits)), k=support)
    pauli_symbols = [rng.choice(["X", "Y", "Z"]) for _ in indices]
    terms = [(_multi_site_pauli(nqubits, indices, pauli_symbols), _rand_coeff(rng))]
    if rng.random() < 0.5:
        indices2 = rng.sample(list(range(nqubits)), k=support)
        pauli_symbols2 = [rng.choice(["X", "Y", "Z"]) for _ in indices2]
        terms.append((_multi_site_pauli(nqubits, indices2, pauli_symbols2), _rand_coeff(rng)))
    return SparsePauliOp.from_list(terms).simplify()


def _single_site_pauli(nqubits: int, index: int, symbol: str) -> str:
    chars = ["I"] * nqubits
    chars[index] = symbol
    return "".join(chars)


def _two_site_pauli(nqubits: int, i: int, j: int, left: str, right: str) -> str:
    chars = ["I"] * nqubits
    chars[i] = left
    chars[j] = right
    return "".join(chars)


def _multi_site_pauli(nqubits: int, indices: list[int], symbols: list[str]) -> str:
    chars = ["I"] * nqubits
    for index, symbol in zip(indices, symbols, strict=True):
        chars[index] = symbol
    return "".join(chars)


def _distinct_pair(nqubits: int, rng: random.Random) -> tuple[int, int]:
    i, j = rng.sample(list(range(nqubits)), k=2)
    return min(i, j), max(i, j)


def _rand_coeff(rng: random.Random) -> float:
    return rng.choice([0.25, 0.5, 0.75, 1.0])

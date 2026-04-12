from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp


CHANNEL_MAP: dict[str, str] = {
    "n": "charge",
    "D": "charge",
    "Ms2": "spin",
    "Cs_max": "spin",
    "K": "transport",
    "Pair_nn": "pair",
    "Pair_span": "pair",
}


@dataclass(frozen=True)
class SuperconductorChannelPlan:
    channels: dict[str, tuple[str, ...]]
    basis_concentration: float
    mean_support: float
    coherence_score: float


def build_superconductor_channel_plan(
    *,
    operator_map: dict[str, SparsePauliOp],
    targets: tuple[str, ...],
) -> SuperconductorChannelPlan:
    grouped: dict[str, list[str]] = defaultdict(list)
    basis_concentrations: list[float] = []
    supports: list[float] = []
    for target in targets:
        grouped[CHANNEL_MAP.get(target, "other")].append(target)
        basis_concentrations.append(_top_basis_share(operator_map[target]))
        supports.append(_mean_support(operator_map[target]))
    basis_concentration = 0.0 if not basis_concentrations else sum(basis_concentrations) / len(basis_concentrations)
    mean_support = 0.0 if not supports else sum(supports) / len(supports)
    # Higher is more coherent: concentrated basis content, low support, fewer channels.
    coherence_score = basis_concentration - 0.15 * max(0.0, mean_support - 2.0) - 0.10 * max(0, len(grouped) - 2)
    return SuperconductorChannelPlan(
        channels={name: tuple(items) for name, items in sorted(grouped.items())},
        basis_concentration=basis_concentration,
        mean_support=mean_support,
        coherence_score=coherence_score,
    )


def _top_basis_share(op: SparsePauliOp) -> float:
    counts = Counter()
    total = 0.0
    for pauli, coeff in op.simplify().to_list():
        weight = abs(coeff)
        for symbol in pauli:
            if symbol != "I":
                counts[symbol] += weight
                total += weight
    if total == 0.0:
        return 1.0
    return counts.most_common(1)[0][1] / total


def _mean_support(op: SparsePauliOp) -> float:
    simplified = op.simplify()
    if len(simplified) == 0:
        return 0.0
    total = 0.0
    for pauli in simplified.paulis:
        total += sum(symbol != "I" for symbol in str(pauli))
    return total / len(simplified)

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from qiskit.quantum_info import SparsePauliOp

from app.physics.measurements import (
    MeasurementGroup,
    group_operator_terms,
    rebuild_operator_from_groups,
)


def transverse_signature(basis: str) -> str:
    return "".join("T" if symbol in {"X", "Y"} else symbol for symbol in basis)


def signature_orbit_key(target_name: str, signature: str) -> str:
    counts = Counter(signature)
    return f"{target_name}|I{counts.get('I', 0)}|Z{counts.get('Z', 0)}|T{counts.get('T', 0)}"


@dataclass(frozen=True)
class DecomposedOperatorBundle:
    operator_map: dict[str, SparsePauliOp]
    target_names: tuple[str, ...]
    avg_signature_groups_per_target: float


def decompose_operator_map_by_transverse_signature(
    *,
    operator_map: dict[str, SparsePauliOp],
    targets: tuple[str, ...],
) -> DecomposedOperatorBundle:
    decomposed: dict[str, SparsePauliOp] = {}
    out_targets: list[str] = []
    signature_counts: list[int] = []
    for target in targets:
        groups = group_operator_terms(target, operator_map[target])
        by_signature: dict[str, list[MeasurementGroup]] = {}
        for group in groups:
            by_signature.setdefault(transverse_signature(group.basis), []).append(group)
        signature_counts.append(len(by_signature))
        for signature, signature_groups in sorted(by_signature.items()):
            name = f"{target}@{signature}"
            decomposed[name] = rebuild_operator_from_groups(signature_groups)
            out_targets.append(name)
    avg_signature_groups = 0.0 if not signature_counts else sum(signature_counts) / len(signature_counts)
    return DecomposedOperatorBundle(
        operator_map=decomposed,
        target_names=tuple(out_targets),
        avg_signature_groups_per_target=avg_signature_groups,
    )

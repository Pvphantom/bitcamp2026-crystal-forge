from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp


MAX_QPROBE_TARGETS_PER_REQUEST = 5
MAX_CUSTOM_OPERATOR_SUPPORT = 6
MAX_CUSTOM_BASIS_FAMILIES = 3
MAX_TOTAL_PAULI_TERMS = 24


@dataclass(frozen=True)
class QProbeBudgetReport:
    num_targets: int
    total_pauli_terms: int
    max_operator_support: int
    basis_families: tuple[str, ...]


def validate_qprobe_request_budget(
    *,
    target_names: tuple[str, ...],
    operator_map: dict[str, SparsePauliOp],
    has_custom_observables: bool,
) -> QProbeBudgetReport:
    if len(target_names) > MAX_QPROBE_TARGETS_PER_REQUEST:
        raise ValueError(
            f"QProbe requests are limited to at most {MAX_QPROBE_TARGETS_PER_REQUEST} target operators per call"
        )

    total_pauli_terms = 0
    max_operator_support = 0
    basis_families: set[str] = set()
    for operator in operator_map.values():
        simplified = operator.simplify()
        total_pauli_terms += len(simplified)
        for pauli in simplified.paulis:
            pauli_str = str(pauli)
            max_operator_support = max(max_operator_support, _pauli_support(pauli_str))
            basis_families.add(_basis_family(pauli_str))

    if has_custom_observables:
        if max_operator_support > MAX_CUSTOM_OPERATOR_SUPPORT:
            raise ValueError(
                f"Custom QProbe operators are limited to support size <= {MAX_CUSTOM_OPERATOR_SUPPORT}; "
                f"received support size {max_operator_support}"
            )
        if len(basis_families) > MAX_CUSTOM_BASIS_FAMILIES:
            raise ValueError(
                f"Custom QProbe requests are limited to at most {MAX_CUSTOM_BASIS_FAMILIES} basis families; "
                f"received {len(basis_families)}"
            )
        if total_pauli_terms > MAX_TOTAL_PAULI_TERMS:
            raise ValueError(
                f"Custom QProbe requests are limited to at most {MAX_TOTAL_PAULI_TERMS} Pauli terms total; "
                f"received {total_pauli_terms}"
            )

    return QProbeBudgetReport(
        num_targets=len(target_names),
        total_pauli_terms=total_pauli_terms,
        max_operator_support=max_operator_support,
        basis_families=tuple(sorted(basis_families)),
    )


def _pauli_support(pauli: str) -> int:
    return sum(1 for symbol in pauli if symbol != "I")


def _basis_family(pauli: str) -> str:
    active = {symbol for symbol in pauli if symbol != "I"}
    if not active or active == {"Z"}:
        return "z"
    if active == {"X"}:
        return "x"
    if active == {"Y"}:
        return "y"
    return "mixed"

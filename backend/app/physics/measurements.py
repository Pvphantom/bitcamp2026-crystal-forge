"""Measurement-group utilities for QProbe.

This module decomposes observables into hardware-relevant Pauli measurement
groups. A group is defined by a basis pattern over qubits (X/Y/Z/I) such that
every Pauli term assigned to that group can be estimated from the same rotated
computational-basis measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp

from app.domain.problem_spec import ProblemSpec
from app.observables.registry import ObservableRegistry, build_default_observable_registry
from app.physics.observables import (
    build_double_occ,
    build_filling,
    build_kinetic,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
)


@dataclass(frozen=True)
class PauliTerm:
    pauli: str
    coeff: complex


@dataclass(frozen=True)
class MeasurementGroup:
    name: str
    basis: str
    terms: tuple[PauliTerm, ...]

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    @property
    def cost(self) -> int:
        # First-order cost proxy: one circuit/basis setting per group.
        return 1

    @property
    def basis_label(self) -> str:
        if set(self.basis) <= {"I", "Z"}:
            return "Charge / spin readout basis"
        if "X" in self.basis and "Y" not in self.basis:
            return "X-sensitive hopping basis"
        if "Y" in self.basis and "X" not in self.basis:
            return "Y-sensitive hopping basis"
        return "Mixed correlation basis"

    @property
    def plain_english(self) -> str:
        if set(self.basis) <= {"I", "Z"}:
            return "Measures charge occupancy and spin-alignment quantities directly in the computational basis."
        if "X" in self.basis and "Y" not in self.basis:
            return "Adds the X-basis information needed to recover motion and hopping-related signals."
        if "Y" in self.basis and "X" not in self.basis:
            return "Adds the Y-basis information needed to recover motion and hopping-related signals."
        return "Captures a mixed set of non-diagonal correlations."


def _canonical_basis_for_term(pauli: str) -> str:
    return "".join("I" if symbol == "I" else symbol for symbol in pauli)


def _basis_compatible(lhs: str, rhs: str) -> bool:
    for left_symbol, right_symbol in zip(lhs, rhs, strict=True):
        if left_symbol == "I" or right_symbol == "I":
            continue
        if left_symbol != right_symbol:
            return False
    return True


def _merge_bases(lhs: str, rhs: str) -> str:
    merged = []
    for left_symbol, right_symbol in zip(lhs, rhs, strict=True):
        if left_symbol == "I":
            merged.append(right_symbol)
        elif right_symbol == "I" or left_symbol == right_symbol:
            merged.append(left_symbol)
        else:
            raise ValueError("Cannot merge incompatible bases")
    return "".join(merged)


def _terms_from_operator(operator: SparsePauliOp) -> list[PauliTerm]:
    simplified = operator.simplify()
    return [
        PauliTerm(pauli=str(pauli), coeff=complex(coeff))
        for pauli, coeff in zip(simplified.paulis, simplified.coeffs, strict=True)
        if abs(coeff) > 1e-12
    ]


def group_operator_terms(name: str, operator: SparsePauliOp) -> list[MeasurementGroup]:
    grouped: list[tuple[str, list[PauliTerm]]] = []
    for term in _terms_from_operator(operator):
        basis = _canonical_basis_for_term(term.pauli)
        placed = False
        for index, (group_basis, terms) in enumerate(grouped):
            if _basis_compatible(group_basis, basis):
                grouped[index] = (_merge_bases(group_basis, basis), [*terms, term])
                placed = True
                break
        if not placed:
            grouped.append((basis, [term]))
    return [
        MeasurementGroup(name=f"{name}:{index}", basis=basis, terms=tuple(terms))
        for index, (basis, terms) in enumerate(sorted(grouped, key=lambda item: item[0]), start=1)
    ]


def rebuild_operator_from_groups(groups: list[MeasurementGroup]) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for group in groups:
        for term in group.terms:
            terms.append((term.pauli, term.coeff))
    return SparsePauliOp.from_list(terms).simplify()


def observable_library(Lx: int, Ly: int, t: float) -> dict[str, SparsePauliOp]:
    return {
        "D": build_double_occ(Lx, Ly),
        "n": build_filling(Lx, Ly),
        "Ms2": build_staggered_magnetization_squared(Lx, Ly),
        "K": build_kinetic(Lx, Ly, t),
        "Cs_max": build_spin_correlator_maxdist(Lx, Ly),
    }


def build_measurement_library(Lx: int, Ly: int, t: float) -> dict[str, list[MeasurementGroup]]:
    return {
        name: group_operator_terms(name, operator)
        for name, operator in observable_library(Lx, Ly, t).items()
    }


def build_measurement_library_from_operator_map(
    operator_map: dict[str, SparsePauliOp],
) -> dict[str, list[MeasurementGroup]]:
    return {
        name: group_operator_terms(name, operator)
        for name, operator in operator_map.items()
    }


def build_measurement_library_for_problem(
    problem: ProblemSpec,
    registry: ObservableRegistry | None = None,
) -> dict[str, list[MeasurementGroup]]:
    observable_registry = registry or build_default_observable_registry()
    return build_measurement_library_from_operator_map(observable_registry.operator_map(problem))


def explain_stop_reason(success: bool, max_uncertainty: float, tolerance: float) -> str:
    if success:
        return (
            "Runtime stop was triggered because the covered targets had uncertainty below the requested tolerance."
        )
    if max_uncertainty > tolerance:
        return "Runtime did not stop early because the uncertainty stayed too large to trust the current plan."
    return "Runtime did not stop early because not all requested targets were covered by the chosen groups."

"""State-preparation helpers following the Hubbard spec conventions."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from app.physics.lattice import q_dn, q_up


OccupationMap = dict[tuple[int, int, str], bool]


def neel_occupations(Lx: int, Ly: int) -> OccupationMap:
    occupations: OccupationMap = {}
    for y in range(Ly):
        for x in range(Lx):
            is_a = (x + y) % 2 == 0
            occupations[(x, y, "up")] = is_a
            occupations[(x, y, "down")] = not is_a
    return occupations


def empty_occupations(Lx: int, Ly: int) -> OccupationMap:
    return {
        (x, y, spin): False
        for y in range(Ly)
        for x in range(Lx)
        for spin in ("up", "down")
    }


def polarized_occupations(Lx: int, Ly: int) -> OccupationMap:
    occupations = empty_occupations(Lx, Ly)
    for y in range(Ly):
        for x in range(Lx):
            occupations[(x, y, "up")] = True
    return occupations


def product_state_occupations(
    Lx: int,
    Ly: int,
    occupations: OccupationMap | None = None,
    *,
    default: str = "empty",
) -> OccupationMap:
    if default == "empty":
        state = empty_occupations(Lx, Ly)
    elif default == "neel":
        state = neel_occupations(Lx, Ly)
    elif default == "polarized":
        state = polarized_occupations(Lx, Ly)
    else:
        raise ValueError("default must be 'empty', 'neel', or 'polarized'")

    if occupations:
        state.update(occupations)
    return state


def occupied_qubits_from_occupations(Lx: int, Ly: int, occupations: OccupationMap) -> list[int]:
    Ns = Lx * Ly
    occupied: list[int] = []
    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            if occupations.get((x, y, "up"), False):
                occupied.append(q_up(i))
            if occupations.get((x, y, "down"), False):
                occupied.append(q_dn(i, Ns))
    return sorted(occupied)


def prepare_product_state_circuit(
    Lx: int,
    Ly: int,
    occupations: OccupationMap | None = None,
    *,
    default: str = "empty",
) -> QuantumCircuit:
    state = product_state_occupations(Lx, Ly, occupations, default=default)
    Nq = 2 * Lx * Ly
    qc = QuantumCircuit(Nq)
    for qubit in occupied_qubits_from_occupations(Lx, Ly, state):
        qc.x(qubit)
    return qc


def prepare_neel_circuit(Lx: int, Ly: int) -> QuantumCircuit:
    return prepare_product_state_circuit(Lx, Ly, default="neel")


def prepare_empty_circuit(Lx: int, Ly: int) -> QuantumCircuit:
    return prepare_product_state_circuit(Lx, Ly, default="empty")


def prepare_polarized_circuit(Lx: int, Ly: int) -> QuantumCircuit:
    return prepare_product_state_circuit(Lx, Ly, default="polarized")


def basis_statevector_from_occupations(
    Lx: int,
    Ly: int,
    occupations: OccupationMap | None = None,
    *,
    default: str = "empty",
) -> np.ndarray:
    Nq = 2 * Lx * Ly
    state = np.zeros(1 << Nq, dtype=complex)
    basis_index = 0
    for qubit in occupied_qubits_from_occupations(
        Lx,
        Ly,
        product_state_occupations(Lx, Ly, occupations, default=default),
    ):
        basis_index |= 1 << qubit
    state[basis_index] = 1.0
    return state

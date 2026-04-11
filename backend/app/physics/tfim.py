"""Transverse-field Ising model builders and observables."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from app.physics.lattice import nn_bonds


def _single_pauli(nqubits: int, qubit: int, symbol: str) -> str:
    pauli = ["I"] * nqubits
    pauli[qubit] = symbol
    return "".join(pauli[::-1])


def _pair_pauli(nqubits: int, q1: int, q2: int, symbol: str) -> str:
    pauli = ["I"] * nqubits
    pauli[q1] = symbol
    pauli[q2] = symbol
    return "".join(pauli[::-1])


def build_tfim_hamiltonian(Lx: int, Ly: int, J: float, h: float, g: float = 0.0) -> SparsePauliOp:
    nsites = Lx * Ly
    terms: list[tuple[str, complex]] = []
    for i, j in nn_bonds(Lx, Ly):
        terms.append((_pair_pauli(nsites, i, j, "Z"), -J))
    for i in range(nsites):
        terms.append((_single_pauli(nsites, i, "X"), -h))
        if abs(g) > 1e-12:
            terms.append((_single_pauli(nsites, i, "Z"), -g))
    return SparsePauliOp.from_list(terms).simplify()


def build_tfim_mz(Lx: int, Ly: int) -> SparsePauliOp:
    nsites = Lx * Ly
    terms = [(_single_pauli(nsites, i, "Z"), 1.0 / nsites) for i in range(nsites)]
    return SparsePauliOp.from_list(terms).simplify()


def build_tfim_mx(Lx: int, Ly: int) -> SparsePauliOp:
    nsites = Lx * Ly
    terms = [(_single_pauli(nsites, i, "X"), 1.0 / nsites) for i in range(nsites)]
    return SparsePauliOp.from_list(terms).simplify()


def build_tfim_zz_nn(Lx: int, Ly: int) -> SparsePauliOp:
    nsites = Lx * Ly
    bonds = nn_bonds(Lx, Ly)
    weight = 1.0 / max(len(bonds), 1)
    terms = [(_pair_pauli(nsites, i, j, "Z"), weight) for i, j in bonds]
    return SparsePauliOp.from_list(terms).simplify()


def build_tfim_staggered_mz(Lx: int, Ly: int) -> SparsePauliOp:
    nsites = Lx * Ly
    terms = []
    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            terms.append((_single_pauli(nsites, i, "Z"), ((-1) ** (x + y)) / nsites))
    return SparsePauliOp.from_list(terms).simplify()


def build_tfim_staggered_mz2(Lx: int, Ly: int) -> SparsePauliOp:
    staggered = build_tfim_staggered_mz(Lx, Ly)
    return SparsePauliOp.from_operator(staggered.to_matrix() @ staggered.to_matrix()).simplify()


def build_tfim_z_span(Lx: int, Ly: int) -> SparsePauliOp:
    nsites = Lx * Ly
    return SparsePauliOp.from_list([(_pair_pauli(nsites, 0, nsites - 1, "Z"), 1.0)]).simplify()


def build_tfim_site_z_operators(Lx: int, Ly: int) -> list[SparsePauliOp]:
    nsites = Lx * Ly
    return [SparsePauliOp.from_list([(_single_pauli(nsites, i, "Z"), 1.0)]).simplify() for i in range(nsites)]


def build_tfim_site_x_operators(Lx: int, Ly: int) -> list[SparsePauliOp]:
    nsites = Lx * Ly
    return [SparsePauliOp.from_list([(_single_pauli(nsites, i, "X"), 1.0)]).simplify() for i in range(nsites)]


def build_tfim_bond_zz_operators(Lx: int, Ly: int) -> dict[tuple[int, int], SparsePauliOp]:
    nsites = Lx * Ly
    return {(i, j): SparsePauliOp.from_list([(_pair_pauli(nsites, i, j, "Z"), 1.0)]).simplify() for i, j in nn_bonds(Lx, Ly)}

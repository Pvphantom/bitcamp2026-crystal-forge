"""Observable builders for the Hubbard model."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from app.physics.lattice import nn_bonds, q_dn, q_up


def _identity_string(Nq: int) -> str:
    return "I" * Nq


def _single_z_string(Nq: int, qubit: int) -> str:
    pauli = ["I"] * Nq
    pauli[qubit] = "Z"
    return "".join(pauli[::-1])


def build_number_operator_for_qubit(Nq: int, qubit: int) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [
            (_identity_string(Nq), 0.5),
            (_single_z_string(Nq, qubit), -0.5),
        ]
    ).simplify()


def build_site_number_operators(Lx: int, Ly: int) -> dict[str, list[SparsePauliOp]]:
    Ns = Lx * Ly
    Nq = 2 * Ns
    n_up = [build_number_operator_for_qubit(Nq, q_up(i)) for i in range(Ns)]
    n_dn = [build_number_operator_for_qubit(Nq, q_dn(i, Ns)) for i in range(Ns)]
    return {"up": n_up, "down": n_dn}


def build_site_double_occupancy_operators(Lx: int, Ly: int) -> list[SparsePauliOp]:
    number_ops = build_site_number_operators(Lx, Ly)
    return [(number_ops["up"][i] @ number_ops["down"][i]).simplify() for i in range(Lx * Ly)]


def build_double_occ(Lx: int, Ly: int) -> SparsePauliOp:
    Ns = Lx * Ly
    return sum(build_site_double_occupancy_operators(Lx, Ly)[1:], build_site_double_occupancy_operators(Lx, Ly)[0]).simplify() / Ns


def build_filling(Lx: int, Ly: int) -> SparsePauliOp:
    Ns = Lx * Ly
    number_ops = build_site_number_operators(Lx, Ly)
    total = sum(number_ops["up"][1:], number_ops["up"][0]).simplify()
    total = (total + sum(number_ops["down"][1:], number_ops["down"][0]).simplify()).simplify()
    return (total / Ns).simplify()


def build_sz_site_operators(Lx: int, Ly: int) -> list[SparsePauliOp]:
    number_ops = build_site_number_operators(Lx, Ly)
    return [((number_ops["up"][i] - number_ops["down"][i]) * 0.5).simplify() for i in range(Lx * Ly)]


def build_staggered_magnetization_squared(Lx: int, Ly: int) -> SparsePauliOp:
    Ns = Lx * Ly
    number_ops = build_site_number_operators(Lx, Ly)
    m_s = None
    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            staggered_density = (number_ops["up"][i] - number_ops["down"][i]).simplify()
            term = staggered_density * (((-1) ** (x + y)) / Ns)
            m_s = term if m_s is None else (m_s + term).simplify()
    assert m_s is not None
    return SparsePauliOp.from_operator(Operator(m_s.to_matrix()) @ Operator(m_s.to_matrix())).simplify()


def build_bond_spin_correlator_operators(Lx: int, Ly: int) -> dict[tuple[int, int], SparsePauliOp]:
    sz_sites = build_sz_site_operators(Lx, Ly)
    return {(i, j): (sz_sites[i] @ sz_sites[j]).simplify() for i, j in nn_bonds(Lx, Ly)}


def build_spin_correlator_maxdist(Lx: int, Ly: int) -> SparsePauliOp:
    bonds = build_bond_spin_correlator_operators(Lx, Ly)
    max_site = Lx * Ly - 1
    if (0, max_site) in bonds:
        return bonds[(0, max_site)]
    sz_sites = build_sz_site_operators(Lx, Ly)
    return (sz_sites[0] @ sz_sites[max_site]).simplify()


def build_kinetic(Lx: int, Ly: int, t: float) -> SparsePauliOp:
    Ns = Lx * Ly
    Nq = 2 * Ns
    bond_terms: list[tuple[str, complex]] = []
    for i, j in nn_bonds(Lx, Ly):
        for spin in ("up", "down"):
            q1 = q_up(i) if spin == "up" else q_dn(i, Ns)
            q2 = q_up(j) if spin == "up" else q_dn(j, Ns)
            q_lo, q_hi = sorted((q1, q2))
            for endpoint in ("X", "Y"):
                pauli = ["I"] * Nq
                pauli[q_lo] = endpoint
                pauli[q_hi] = endpoint
                for q in range(q_lo + 1, q_hi):
                    pauli[q] = "Z"
                bond_terms.append(("".join(pauli[::-1]), -0.5 * t / len(nn_bonds(Lx, Ly))))
    return SparsePauliOp.from_list(bond_terms).simplify()


def extract_site_observables_from_statevector(Lx: int, Ly: int, state: np.ndarray) -> dict[str, list[float]]:
    """Direct basis-probability evaluation for diagonal site observables."""
    Ns = Lx * Ly
    probs = np.abs(state) ** 2
    n_up = [0.0] * Ns
    n_dn = [0.0] * Ns
    d_site = [0.0] * Ns
    sz_site = [0.0] * Ns

    for basis_index, prob in enumerate(probs):
        for i in range(Ns):
            up = (basis_index >> q_up(i)) & 1
            dn = (basis_index >> q_dn(i, Ns)) & 1
            n_up[i] += prob * up
            n_dn[i] += prob * dn
            d_site[i] += prob * up * dn
            sz_site[i] += prob * 0.5 * (up - dn)

    return {
        "n_up": n_up,
        "n_dn": n_dn,
        "D_site": d_site,
        "Sz_site": sz_site,
    }


def extract_global_observables_from_statevector(Lx: int, Ly: int, state: np.ndarray) -> dict[str, float]:
    site = extract_site_observables_from_statevector(Lx, Ly, state)
    Ns = Lx * Ly

    d = float(sum(site["D_site"]) / Ns)
    filling = float(sum(site["n_up"]) + sum(site["n_dn"])) / Ns

    staggered = 0.0
    probs = np.abs(state) ** 2
    for basis_index, prob in enumerate(probs):
        ms = 0.0
        for y in range(Ly):
            for x in range(Lx):
                i = x + Lx * y
                up = (basis_index >> q_up(i)) & 1
                dn = (basis_index >> q_dn(i, Ns)) & 1
                ms += ((-1) ** (x + y)) * (up - dn) / Ns
        staggered += prob * (ms ** 2)

    return {"D": d, "n": filling, "Ms2": float(staggered)}

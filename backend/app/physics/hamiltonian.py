r"""Hubbard Hamiltonian builders.

This module follows the project reference conventions:

- spin-major qubit ordering: q(i, up) = i, q(i, down) = Ns + i
- Qiskit SparsePauliOp strings are little-endian, so any list-indexed Pauli
  construction must reverse with [::-1] before joining
- the primary builder uses qiskit_nature, but permutes the fermionic mode
  indices into the project's spin-major ordering before applying the
  Jordan-Wigner mapper
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, SquareLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

from app.physics.lattice import nn_bonds, q_dn, q_up, site_major_to_spin_major_layout


def build_hamiltonian(Lx: int, Ly: int, t: float, U: float, mu: float) -> SparsePauliOp:
    """Build the Fermi-Hubbard Hamiltonian in the project's spin-major ordering."""
    lattice = SquareLattice(rows=Ly, cols=Lx, boundary_condition=BoundaryCondition.OPEN)
    parameters = lattice.uniform_parameters(
        uniform_interaction=-t,
        uniform_onsite_potential=-mu,
    )
    model = FermiHubbardModel(parameters, onsite_interaction=U)
    Ns = Lx * Ly
    fermionic_op = model.second_q_op().permute_indices(site_major_to_spin_major_layout(Ns))
    return JordanWignerMapper().map(fermionic_op).simplify()


def build_number_operator(Lx: int, Ly: int, spin: str | None = None) -> SparsePauliOp:
    """Build N_up, N_down, or N_total in spin-major ordering."""
    Ns = Lx * Ly
    Nq = 2 * Ns
    terms: list[tuple[str, complex]] = []

    if spin == "up":
        qubits = [q_up(i) for i in range(Ns)]
    elif spin == "down":
        qubits = [q_dn(i, Ns) for i in range(Ns)]
    elif spin is None:
        qubits = list(range(Nq))
    else:
        raise ValueError("spin must be 'up', 'down', or None")

    for qubit in qubits:
        identity = ["I"] * Nq
        z_term = ["I"] * Nq
        z_term[qubit] = "Z"
        terms.append(("".join(identity[::-1]), 0.5))
        terms.append(("".join(z_term[::-1]), -0.5))

    return SparsePauliOp.from_list(terms).simplify()


def build_hamiltonian_manual(Lx: int, Ly: int, t: float, U: float, mu: float) -> SparsePauliOp:
    r"""Independent manual builder used as a physics cross-check.

    Sign convention:
    - The hopping piece is -t * (c_i^\dagger c_j + c_j^\dagger c_i)
    - Under the Jordan-Wigner convention used by qiskit_nature, the Hermitian
      hopping combination maps to +(1/2)(X Z... X + Y Z... Y)
    - Therefore each Pauli hopping term carries coefficient -t/2
    """
    Ns = Lx * Ly
    Nq = 2 * Ns
    terms: list[tuple[str, complex]] = []

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
                terms.append(("".join(pauli[::-1]), -0.5 * t))

    for i in range(Ns):
        qu = q_up(i)
        qd = q_dn(i, Ns)

        identity = ["I"] * Nq
        zu = ["I"] * Nq
        zd = ["I"] * Nq
        zudz = ["I"] * Nq
        zu[qu] = "Z"
        zd[qd] = "Z"
        zudz[qu] = "Z"
        zudz[qd] = "Z"

        terms.extend(
            [
                ("".join(identity[::-1]), U / 4.0),
                ("".join(zu[::-1]), -U / 4.0),
                ("".join(zd[::-1]), -U / 4.0),
                ("".join(zudz[::-1]), U / 4.0),
            ]
        )

    identity = ["I"] * Nq
    for q in range(Nq):
        z_term = ["I"] * Nq
        z_term[q] = "Z"
        terms.append(("".join(identity[::-1]), -mu / 2.0))
        terms.append(("".join(z_term[::-1]), mu / 2.0))

    return SparsePauliOp.from_list(terms).simplify()

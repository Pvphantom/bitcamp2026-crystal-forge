"""Lattice helpers following the Hubbard spec's spin-major ordering.

Qubit convention:
- q(i, up) = i
- q(i, down) = Ns + i

Qiskit Nature currently emits the Fermi-Hubbard spin orbitals in site-major
order for this model: [0_up, 0_dn, 1_up, 1_dn, ...]. The helper
``site_major_to_spin_major_layout`` provides the explicit permutation into the
project's required spin-major ordering.
"""


def site_index(x: int, y: int, Lx: int) -> int:
    return x + Lx * y


def q_up(i: int) -> int:
    return i


def q_dn(i: int, Ns: int) -> int:
    return Ns + i


def nn_bonds(Lx: int, Ly: int) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for y in range(Ly):
        for x in range(Lx):
            i = site_index(x, y, Lx)
            if x + 1 < Lx:
                bonds.append((i, i + 1))
            if y + 1 < Ly:
                bonds.append((i, i + Lx))
    return bonds


def site_major_to_spin_major_layout(Ns: int) -> list[int]:
    """Return a Qiskit layout list mapping old site-major qubits to new spin-major qubits."""
    layout = [0] * (2 * Ns)
    for i in range(Ns):
        layout[2 * i] = q_up(i)
        layout[2 * i + 1] = q_dn(i, Ns)
    return layout

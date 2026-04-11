import numpy as np

from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.observables import (
    build_bond_spin_correlator_operators,
    build_double_occ,
    build_filling,
    build_kinetic,
    build_site_double_occupancy_operators,
    build_site_number_operators,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
    extract_global_observables_from_statevector,
    extract_site_observables_from_statevector,
)
from app.physics.state_prep import basis_statevector_from_occupations


def test_diagonal_observables_match_direct_statevector_extraction() -> None:
    h_op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    _, psi0 = ground_state(h_op)

    site_direct = extract_site_observables_from_statevector(2, 2, psi0)
    global_direct = extract_global_observables_from_statevector(2, 2, psi0)

    number_ops = build_site_number_operators(2, 2)
    double_ops = build_site_double_occupancy_operators(2, 2)

    for i in range(4):
        assert abs(expectation_value(number_ops["up"][i], psi0) - site_direct["n_up"][i]) < 1e-10
        assert abs(expectation_value(number_ops["down"][i], psi0) - site_direct["n_dn"][i]) < 1e-10
        assert abs(expectation_value(double_ops[i], psi0) - site_direct["D_site"][i]) < 1e-10

    assert abs(expectation_value(build_double_occ(2, 2), psi0) - global_direct["D"]) < 1e-10
    assert abs(expectation_value(build_filling(2, 2), psi0) - global_direct["n"]) < 1e-10
    assert abs(expectation_value(build_staggered_magnetization_squared(2, 2), psi0) - global_direct["Ms2"]) < 1e-10


def test_observable_physical_sanity_checks() -> None:
    psi_neel = basis_statevector_from_occupations(2, 2, default="neel")
    assert abs(expectation_value(build_staggered_magnetization_squared(2, 2), psi_neel) - 1.0) < 1e-10

    _, psi_u0 = ground_state(build_hamiltonian(2, 2, t=1.0, U=0.0, mu=0.0))
    d_u0 = expectation_value(build_double_occ(2, 2), psi_u0)
    assert 0.15 < d_u0 < 0.30
    assert expectation_value(build_kinetic(2, 2, t=1.0), psi_u0) < 0.0

    _, psi_u20 = ground_state(build_hamiltonian(2, 2, t=1.0, U=20.0, mu=10.0))
    d_u20 = expectation_value(build_double_occ(2, 2), psi_u20)
    assert d_u20 < 0.05

    _, psi_half = ground_state(build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0))
    filling = expectation_value(build_filling(2, 2), psi_half)
    assert abs(filling - 1.0) < 0.01


def test_spin_correlator_builders_are_hermitian_and_bond_complete() -> None:
    bond_ops = build_bond_spin_correlator_operators(2, 2)
    assert set(bond_ops.keys()) == {(0, 1), (0, 2), (1, 3), (2, 3)}
    maxdist = build_spin_correlator_maxdist(2, 2)
    matrix = maxdist.to_matrix()
    assert np.allclose(matrix, matrix.conj().T, atol=1e-10)

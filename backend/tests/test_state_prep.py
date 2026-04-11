import numpy as np

from app.physics.state_prep import (
    basis_statevector_from_occupations,
    neel_occupations,
    occupied_qubits_from_occupations,
    prepare_empty_circuit,
    prepare_neel_circuit,
    prepare_polarized_circuit,
    product_state_occupations,
)


def test_neel_occupations_match_sublattice_pattern() -> None:
    occ = neel_occupations(2, 2)
    assert occ[(0, 0, "up")] is True
    assert occ[(0, 0, "down")] is False
    assert occ[(1, 0, "up")] is False
    assert occ[(1, 0, "down")] is True
    assert occ[(0, 1, "up")] is False
    assert occ[(0, 1, "down")] is True
    assert occ[(1, 1, "up")] is True
    assert occ[(1, 1, "down")] is False


def test_product_state_overrides_default_configuration() -> None:
    occ = product_state_occupations(2, 2, {(0, 0, "down"): True}, default="neel")
    assert occ[(0, 0, "up")] is True
    assert occ[(0, 0, "down")] is True


def test_occupied_qubits_are_spin_major() -> None:
    occ = neel_occupations(2, 2)
    assert occupied_qubits_from_occupations(2, 2, occ) == [0, 3, 5, 6]


def test_basis_statevector_for_neel_is_single_basis_state() -> None:
    state = basis_statevector_from_occupations(2, 2, default="neel")
    assert abs(np.vdot(state, state).real - 1.0) < 1e-12
    assert np.count_nonzero(np.abs(state) > 0) == 1


def test_circuit_builders_have_expected_x_counts() -> None:
    assert prepare_empty_circuit(2, 2).count_ops().get("x", 0) == 0
    assert prepare_neel_circuit(2, 2).count_ops().get("x", 0) == 4
    assert prepare_polarized_circuit(2, 2).count_ops().get("x", 0) == 4

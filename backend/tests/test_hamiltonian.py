from __future__ import annotations

import numpy as np

from app.physics.hamiltonian import (
    build_hamiltonian,
    build_hamiltonian_manual,
    build_number_operator,
)


def _ground_energy(op) -> float:
    matrix = op.to_matrix()
    eigvals = np.linalg.eigvalsh(matrix)
    return float(np.min(eigvals).real)


def _neel_statevector_2x2() -> np.ndarray:
    """Spin-major Neel state for a 2x2 lattice.

    Sites:
    - 0 (0,0): up occupied
    - 1 (1,0): down occupied
    - 2 (0,1): down occupied
    - 3 (1,1): up occupied
    """
    nq = 8
    occupied_qubits = {0, 3, 5, 6}
    basis_index = sum(1 << q for q in occupied_qubits)
    state = np.zeros(1 << nq, dtype=complex)
    state[basis_index] = 1.0
    return state


def _occupation_expectation(qubit: int, state: np.ndarray) -> float:
    nq = int(np.log2(state.shape[0]))
    probs = np.abs(state) ** 2
    total = 0.0
    for index, prob in enumerate(probs):
        total += prob * ((index >> qubit) & 1)
    return total


def test_hamiltonian_manual_matches_qiskit_nature_builder() -> None:
    h_ours = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0).to_matrix()
    h_manual = build_hamiltonian_manual(2, 2, t=1.0, U=4.0, mu=2.0).to_matrix()
    assert np.allclose(h_ours, h_manual, atol=1e-10)


def test_number_conservation_for_each_spin_sector() -> None:
    h_matrix = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0).to_matrix()
    n_up = build_number_operator(2, 2, spin="up").to_matrix()
    n_down = build_number_operator(2, 2, spin="down").to_matrix()

    comm_up = h_matrix @ n_up - n_up @ h_matrix
    comm_down = h_matrix @ n_down - n_down @ h_matrix

    assert np.max(np.abs(comm_up)) < 1e-10
    assert np.max(np.abs(comm_down)) < 1e-10


def test_free_fermion_ground_state_energy_is_minus_four() -> None:
    energy = _ground_energy(build_hamiltonian(2, 2, t=1.0, U=0.0, mu=0.0))
    assert abs(energy - (-4.0)) < 1e-10


def test_particle_hole_symmetry_derivative_at_half_filling() -> None:
    delta = 1e-3
    e_mu = _ground_energy(build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0))
    e_mu_delta = _ground_energy(build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0 + delta))
    derivative = -(e_mu_delta - e_mu) / delta
    assert abs(derivative - 4.0) < 1e-2


def test_randomized_parameter_sweeps_preserve_manual_builder_agreement() -> None:
    parameter_points = [
        (1.0, 0.0, 0.0),
        (1.0, 4.0, 2.0),
        (0.7, 3.5, 1.2),
        (1.3, 6.0, 2.8),
    ]
    for t, U, mu in parameter_points:
        h_ours = build_hamiltonian(2, 2, t=t, U=U, mu=mu).to_matrix()
        h_manual = build_hamiltonian_manual(2, 2, t=t, U=U, mu=mu).to_matrix()
        assert np.allclose(h_ours, h_manual, atol=1e-10)


def test_builder_is_hermitian_and_matches_manual_spectrum() -> None:
    h_ours = build_hamiltonian(2, 2, t=0.9, U=5.0, mu=1.7).to_matrix()
    h_manual = build_hamiltonian_manual(2, 2, t=0.9, U=5.0, mu=1.7).to_matrix()

    assert np.allclose(h_ours, h_ours.conj().T, atol=1e-10)
    assert np.allclose(h_manual, h_manual.conj().T, atol=1e-10)

    evals_ours = np.linalg.eigvalsh(h_ours)
    evals_manual = np.linalg.eigvalsh(h_manual)
    assert np.allclose(evals_ours, evals_manual, atol=1e-10)


def test_neel_state_ordering_matches_spin_major_convention() -> None:
    state = _neel_statevector_2x2()

    assert _occupation_expectation(0, state) == 1.0
    assert _occupation_expectation(4, state) == 0.0

    assert _occupation_expectation(1, state) == 0.0
    assert _occupation_expectation(5, state) == 1.0

    assert _occupation_expectation(2, state) == 0.0
    assert _occupation_expectation(6, state) == 1.0

    assert _occupation_expectation(3, state) == 1.0
    assert _occupation_expectation(7, state) == 0.0

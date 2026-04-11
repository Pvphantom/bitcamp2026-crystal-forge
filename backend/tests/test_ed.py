import numpy as np

from app.physics.ed import diagonalize, expectation_value, ground_state, is_hermitian
from app.physics.hamiltonian import build_hamiltonian


def test_exact_diagonalization_returns_sorted_real_spectrum() -> None:
    op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    eigvals, eigvecs = diagonalize(op)
    assert np.all(np.diff(eigvals) >= -1e-12)
    assert eigvecs.shape == (256, 256)


def test_ground_state_matches_lowest_eigenvalue() -> None:
    op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    eigvals, _ = diagonalize(op)
    e0, psi0 = ground_state(op)
    assert abs(e0 - eigvals[0]) < 1e-12
    assert abs(np.vdot(psi0, psi0).real - 1.0) < 1e-12


def test_ed_operator_is_hermitian_and_energy_expectation_is_consistent() -> None:
    op = build_hamiltonian(2, 2, t=0.8, U=3.0, mu=1.1)
    e0, psi0 = ground_state(op)
    assert is_hermitian(op)
    assert abs(expectation_value(op, psi0) - e0) < 1e-10

"""Exact diagonalization utilities for small Hubbard lattices."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse.linalg import eigsh


def operator_matrix(op: SparsePauliOp) -> np.ndarray:
    """Return the dense matrix representation of an operator."""
    return np.asarray(op.to_matrix(), dtype=complex)


def is_hermitian(op: SparsePauliOp, atol: float = 1e-10) -> bool:
    matrix = operator_matrix(op)
    return np.allclose(matrix, matrix.conj().T, atol=atol)


def diagonalize(op: SparsePauliOp) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a Hermitian operator exactly."""
    matrix = operator_matrix(op)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvals.real, eigvecs


def ground_state(op: SparsePauliOp) -> tuple[float, np.ndarray]:
    """Return the ground-state energy and normalized statevector."""
    if op.num_qubits > 8:
        sparse = op.to_matrix(sparse=True)
        eigvals, eigvecs = eigsh(sparse, k=1, which="SA")
        state = eigvecs[:, 0]
        state = state / np.linalg.norm(state)
        return float(eigvals[0].real), state

    matrix = operator_matrix(op)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return float(eigvals[0].real), eigvecs[:, 0]


def expectation_value(op: SparsePauliOp, state: np.ndarray) -> float:
    """Compute <state|op|state> for a normalized statevector."""
    matrix = operator_matrix(op)
    return float(np.vdot(state, matrix @ state).real)

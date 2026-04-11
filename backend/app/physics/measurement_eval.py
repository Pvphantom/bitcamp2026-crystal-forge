"""Shot-based evaluation of grouped Pauli measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.physics.ed import expectation_value
from app.physics.measurements import MeasurementGroup, observable_library


@dataclass(frozen=True)
class NoiseModel:
    readout_flip_prob: float = 0.0


def _apply_basis_rotation(state: np.ndarray, basis: str) -> np.ndarray:
    rotated = np.asarray(state, dtype=complex)
    n_qubits = int(np.log2(rotated.shape[0]))
    for qubit, symbol in enumerate(reversed(basis)):
        if symbol == "I" or symbol == "Z":
            continue
        gate = None
        if symbol == "X":
            gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
        elif symbol == "Y":
            gate = (
                np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2.0)
            )
        else:
            raise ValueError(f"Unsupported basis symbol {symbol}")
        rotated = _apply_single_qubit_gate(rotated, gate, qubit, n_qubits)
    return rotated


def _apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    axis = n_qubits - 1 - qubit
    tensor = np.moveaxis(tensor, axis, 0)
    tensor = np.tensordot(gate, tensor, axes=[[1], [0]])
    tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def _sample_measurement_outcomes(
    state: np.ndarray,
    basis: str,
    shots: int,
    rng: np.random.Generator,
    noise_model: NoiseModel,
) -> np.ndarray:
    rotated = _apply_basis_rotation(state, basis)
    probabilities = np.abs(rotated) ** 2
    samples = rng.choice(len(probabilities), size=shots, p=probabilities)
    bits = ((samples[:, None] >> np.arange(len(basis))) & 1).astype(np.int8)
    if noise_model.readout_flip_prob > 0.0:
        flips = rng.random(bits.shape) < noise_model.readout_flip_prob
        measured_mask = np.array([symbol != "I" for symbol in reversed(basis)], dtype=bool)
        flips &= measured_mask[None, :]
        bits = np.bitwise_xor(bits, flips.astype(np.int8))
    return bits


def _estimate_term_from_samples(pauli: str, bits: np.ndarray) -> float:
    active_qubits = [index for index, symbol in enumerate(reversed(pauli)) if symbol != "I"]
    if not active_qubits:
        return 1.0
    parity = bits[:, active_qubits].sum(axis=1) % 2
    eigenvalues = 1.0 - 2.0 * parity
    return float(np.mean(eigenvalues))


def evaluate_measurement_groups(
    state: np.ndarray,
    groups: list[MeasurementGroup],
    *,
    shots_per_group: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(seed)
    noise = noise_model or NoiseModel()
    estimates: dict[str, float] = {}
    term_expectations: dict[str, float] = {}

    for group in groups:
        bits = _sample_measurement_outcomes(state, group.basis, shots_per_group, rng, noise)
        subtotal = 0.0
        for term in group.terms:
            term_estimate = _estimate_term_from_samples(term.pauli, bits)
            term_expectations[term.pauli] = term_estimate
            subtotal += float(term.coeff.real) * term_estimate
        observable_name = group.name.split(":")[0]
        estimates[observable_name] = estimates.get(observable_name, 0.0) + subtotal
    return estimates, term_expectations


def evaluate_observable_library(
    state: np.ndarray,
    measurement_library: dict[str, list[MeasurementGroup]],
    *,
    shots_per_group: int,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    all_groups = [group for groups in measurement_library.values() for group in groups]
    estimated, _ = evaluate_measurement_groups(
        state,
        all_groups,
        shots_per_group=shots_per_group,
        noise_model=noise_model,
        seed=seed,
    )

    exact = {}
    errors = {}
    for name, operator in observable_library_from_groups(measurement_library).items():
        exact[name] = expectation_value(operator, state)
        errors[name] = abs(estimated.get(name, 0.0) - exact[name])
    return {"estimated": estimated, "exact": exact, "abs_error": errors}


def observable_library_from_groups(
    measurement_library: dict[str, list[MeasurementGroup]],
) -> dict[str, object]:
    from app.physics.measurements import rebuild_operator_from_groups

    return {
        name: rebuild_operator_from_groups(groups)
        for name, groups in measurement_library.items()
    }


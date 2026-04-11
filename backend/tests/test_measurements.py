import numpy as np

from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel, evaluate_observable_library
from app.physics.measurements import (
    build_measurement_library,
    observable_library,
    rebuild_operator_from_groups,
)
from app.physics.state_prep import basis_statevector_from_occupations


def test_measurement_groups_rebuild_each_observable_exactly() -> None:
    library = build_measurement_library(2, 2, t=1.0)
    operators = observable_library(2, 2, t=1.0)
    for name, groups in library.items():
        rebuilt = rebuild_operator_from_groups(groups)
        original = operators[name]
        assert np.allclose(rebuilt.to_matrix(), original.to_matrix(), atol=1e-10)


def test_measurement_groups_cover_nonzero_pauli_terms_without_loss() -> None:
    library = build_measurement_library(2, 2, t=1.0)
    operators = observable_library(2, 2, t=1.0)
    for name, groups in library.items():
        grouped_terms = sum(group.num_terms for group in groups)
        original_terms = len(operators[name].simplify().coeffs)
        assert grouped_terms == original_terms


def test_measurement_estimates_converge_to_exact_values_in_zero_noise_limit() -> None:
    state = basis_statevector_from_occupations(2, 2, default="neel")
    library = build_measurement_library(2, 2, t=1.0)
    result = evaluate_observable_library(
        state,
        library,
        shots_per_group=20000,
        noise_model=NoiseModel(readout_flip_prob=0.0),
        seed=7,
    )
    for name, error in result["abs_error"].items():
        assert error < 0.03, f"{name} error too large: {error}"


def test_more_shots_reduce_average_measurement_error() -> None:
    h_op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    _, state = ground_state(h_op)
    library = build_measurement_library(2, 2, t=1.0)

    low_errors = []
    high_errors = []
    for seed in range(10):
        low = evaluate_observable_library(
            state,
            library,
            shots_per_group=200,
            noise_model=NoiseModel(readout_flip_prob=0.0),
            seed=seed,
        )
        high = evaluate_observable_library(
            state,
            library,
            shots_per_group=4000,
            noise_model=NoiseModel(readout_flip_prob=0.0),
            seed=seed,
        )
        low_errors.append(np.mean(list(low["abs_error"].values())))
        high_errors.append(np.mean(list(high["abs_error"].values())))
    assert float(np.mean(high_errors)) < float(np.mean(low_errors))


def test_readout_noise_worsens_average_reconstruction_error() -> None:
    h_op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    _, state = ground_state(h_op)
    library = build_measurement_library(2, 2, t=1.0)

    clean_errors = []
    noisy_errors = []
    for seed in range(10):
        clean = evaluate_observable_library(
            state,
            library,
            shots_per_group=2000,
            noise_model=NoiseModel(readout_flip_prob=0.0),
            seed=seed,
        )
        noisy = evaluate_observable_library(
            state,
            library,
            shots_per_group=2000,
            noise_model=NoiseModel(readout_flip_prob=0.08),
            seed=seed,
        )
        clean_errors.append(np.mean(list(clean["abs_error"].values())))
        noisy_errors.append(np.mean(list(noisy["abs_error"].values())))
    assert float(np.mean(noisy_errors)) > float(np.mean(clean_errors))


def test_group_reconstruction_matches_exact_expectation_operator_by_operator() -> None:
    h_op = build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0)
    _, state = ground_state(h_op)
    library = build_measurement_library(2, 2, t=1.0)
    operators = observable_library(2, 2, t=1.0)
    for name, groups in library.items():
        rebuilt = rebuild_operator_from_groups(groups)
        assert abs(expectation_value(rebuilt, state) - expectation_value(operators[name], state)) < 1e-10

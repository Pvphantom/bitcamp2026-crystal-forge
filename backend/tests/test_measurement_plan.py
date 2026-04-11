from app.optimization.measurement_plan import search_minimal_measurement_plan
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.state_prep import basis_statevector_from_occupations


def test_z_only_targets_reduce_to_single_measurement_group() -> None:
    state = basis_statevector_from_occupations(2, 2, default="neel")
    result = search_minimal_measurement_plan(
        Lx=2,
        Ly=2,
        t=1.0,
        state=state,
        target_observables=("D", "n", "Ms2", "Cs_max"),
        tolerance=0.03,
        shots_per_group=20000,
        noise_model=NoiseModel(readout_flip_prob=0.0),
        seed=11,
    )
    assert result.success is True
    assert result.recommended_plan.cost == 1
    assert result.recommended_plan.cost <= result.full_plan.cost
    assert result.max_abs_error <= result.tolerance


def test_kinetic_target_requires_four_measurement_groups() -> None:
    _, state = ground_state(build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0))
    result = search_minimal_measurement_plan(
        Lx=2,
        Ly=2,
        t=1.0,
        state=state,
        target_observables=("K",),
        tolerance=0.05,
        shots_per_group=40000,
        noise_model=NoiseModel(readout_flip_prob=0.0),
        seed=3,
    )
    assert result.success is True
    assert result.recommended_plan.cost == 4
    assert result.max_abs_error <= result.tolerance


def test_impossible_tolerance_returns_full_plan_with_failure() -> None:
    state = basis_statevector_from_occupations(2, 2, default="neel")
    result = search_minimal_measurement_plan(
        Lx=2,
        Ly=2,
        t=1.0,
        state=state,
        target_observables=("D", "n", "Ms2", "K", "Cs_max"),
        tolerance=1e-4,
        shots_per_group=500,
        noise_model=NoiseModel(readout_flip_prob=0.08),
        seed=21,
    )
    assert result.success is False
    assert result.recommended_plan.cost == result.full_plan.cost
    assert result.max_abs_error > result.tolerance

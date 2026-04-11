from app.optimization.measurement_plan import search_minimal_measurement_plan
from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel
from app.physics.observables import (
    build_double_occ,
    build_filling,
    build_kinetic,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
)
from scripts.data_gen_qprobe import build_feature_vector


def test_qprobe_feature_vector_shape_is_stable() -> None:
    features = build_feature_vector(
        U=4.0,
        mu=2.0,
        D=0.1,
        n=1.0,
        Ms2=0.5,
        K=-0.7,
        Cs_max=-0.1,
        tolerance=0.03,
        shots_per_group=4000,
        readout_flip_prob=0.02,
        targets=("D", "n", "Ms2", "Cs_max"),
    )
    assert features.shape == (15,)


def test_qprobe_oracle_labels_have_consistent_cost_relationship() -> None:
    h_op = build_hamiltonian(2, 2, t=1.0, U=8.0, mu=4.0)
    _, state = ground_state(h_op)
    D = expectation_value(build_double_occ(2, 2), state)
    n = expectation_value(build_filling(2, 2), state)
    Ms2 = expectation_value(build_staggered_magnetization_squared(2, 2), state)
    K = expectation_value(build_kinetic(2, 2, t=1.0), state)
    Cs_max = expectation_value(build_spin_correlator_maxdist(2, 2), state)

    result = search_minimal_measurement_plan(
        Lx=2,
        Ly=2,
        t=1.0,
        state=state,
        target_observables=("D", "n", "Ms2", "Cs_max"),
        tolerance=0.03,
        shots_per_group=4000,
        noise_model=NoiseModel(readout_flip_prob=0.02),
        seed=11,
    )
    features = build_feature_vector(
        U=8.0,
        mu=4.0,
        D=D,
        n=n,
        Ms2=Ms2,
        K=K,
        Cs_max=Cs_max,
        tolerance=0.03,
        shots_per_group=4000,
        readout_flip_prob=0.02,
        targets=("D", "n", "Ms2", "Cs_max"),
    )
    assert features[0].item() == 8.0
    assert result.recommended_plan.cost <= result.full_plan.cost
    assert result.max_abs_error <= 0.03

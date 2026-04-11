from app.optimization.measurement_plan import search_adaptive_measurement_plan
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel


def test_adaptive_qprobe_stops_early_on_compression_case() -> None:
    _, state = ground_state(build_hamiltonian(2, 2, t=1.0, U=8.0, mu=4.0))
    result = search_adaptive_measurement_plan(
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
    assert result.success is True
    assert result.final_plan.cost < result.full_plan.cost
    assert len(result.steps) == result.final_plan.cost
    assert result.max_abs_error <= result.tolerance
    assert result.max_uncertainty <= result.tolerance


def test_adaptive_qprobe_uses_full_plan_on_hard_case() -> None:
    _, state = ground_state(build_hamiltonian(2, 2, t=1.0, U=4.0, mu=2.0))
    result = search_adaptive_measurement_plan(
        Lx=2,
        Ly=2,
        t=1.0,
        state=state,
        target_observables=("D", "n", "Ms2", "K", "Cs_max"),
        tolerance=0.01,
        shots_per_group=2000,
        noise_model=NoiseModel(readout_flip_prob=0.08),
        seed=11,
    )
    assert result.success is False
    assert result.final_plan.cost == result.full_plan.cost
    assert len(result.steps) == result.full_plan.cost

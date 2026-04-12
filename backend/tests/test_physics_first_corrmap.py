from app.analysis.general_tractability_features import analyze_general_tractability_features
from app.analysis.physics_first_corrmap import PhysicsFirstCorrMapConfig, score_physics_first_corrmap
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap
from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver


def test_physics_first_corrmap_prefers_mean_field_in_easy_hubbard_regime() -> None:
    problem = ProblemSpec.hubbard(Lx=6, Ly=6, t=1.0, U=1.0, mu=4.0)
    cheap = MeanFieldSolver().solve(problem)
    runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
    general = analyze_general_tractability_features(
        cheap_result=cheap,
        stability=runtime.stability,
        sensitivity=runtime.sensitivity,
        size_consistency=runtime.size_consistency,
        ansatz_disagreement=runtime.ansatz_disagreement,
        hysteresis=runtime.hysteresis,
        physical_tractability=runtime.physical_tractability,
    )
    report = score_physics_first_corrmap(runtime=runtime, general=general, config=PhysicsFirstCorrMapConfig())
    assert report.label in {"mean_field", "scalable_classical"}
    assert report.mean_field_score >= report.quantum_frontier_score


def test_physics_first_corrmap_prefers_quantum_in_critical_tfim_regime() -> None:
    problem = ProblemSpec.tfim(Lx=6, Ly=6, J=1.0, h=1.0, g=0.0)
    cheap = TFIMMeanFieldSolver().solve(problem)
    runtime = analyze_runtime_intrinsic_corrmap(problem, cheap_result=cheap)
    general = analyze_general_tractability_features(
        cheap_result=cheap,
        stability=runtime.stability,
        sensitivity=runtime.sensitivity,
        size_consistency=runtime.size_consistency,
        ansatz_disagreement=runtime.ansatz_disagreement,
        hysteresis=runtime.hysteresis,
        physical_tractability=runtime.physical_tractability,
    )
    report = score_physics_first_corrmap(runtime=runtime, general=general, config=PhysicsFirstCorrMapConfig())
    assert report.quantum_frontier_score >= report.mean_field_score

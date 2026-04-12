from app.analysis.intrinsic_risk import assess_intrinsic_risk
from app.analysis.mf_ansatz_disagreement import analyze_mean_field_ansatz_disagreement
from app.analysis.mf_hysteresis import analyze_mean_field_hysteresis
from app.analysis.mf_sensitivity import analyze_mean_field_sensitivity
from app.analysis.mf_size_consistency import analyze_mean_field_size_consistency
from app.analysis.mf_stability import analyze_mean_field_stability
from app.analysis.runtime_intrinsic_corrmap import analyze_runtime_intrinsic_corrmap, apply_runtime_intrinsic_overlay
from app.domain.problem_spec import ProblemSpec
from app.solvers.mean_field import MeanFieldSettings, MeanFieldSolver
from app.solvers.tfim_mean_field import TFIMMeanFieldSettings, TFIMMeanFieldSolver
from scripts.data_gen_intrinsic_corrmap import build_intrinsic_sample


def test_mean_field_solver_emits_intrinsic_diagnostics() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    result = MeanFieldSolver(MeanFieldSettings(seed=3, init_noise_scale=0.02)).solve(problem)
    assert "residual_norm" in result.metadata
    assert "final_delta" in result.metadata
    assert result.metadata["seed"] == 3


def test_tfim_mean_field_solver_emits_intrinsic_diagnostics() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    result = TFIMMeanFieldSolver(TFIMMeanFieldSettings(seed=5, init_noise_scale=0.02)).solve(problem)
    assert "residual_norm" in result.metadata
    assert "final_delta" in result.metadata
    assert result.metadata["seed"] == 5


def test_stability_and_sensitivity_reports_are_well_formed() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    stability = analyze_mean_field_stability(problem, num_seeds=3, init_noise_scale=0.02)
    sensitivity = analyze_mean_field_sensitivity(problem, perturbation_scale=0.05)
    size_consistency = analyze_mean_field_size_consistency(problem)
    ansatz_disagreement = analyze_mean_field_ansatz_disagreement(problem)
    hysteresis = analyze_mean_field_hysteresis(problem, perturbation_scale=0.05)
    assessment = assess_intrinsic_risk(
        stability=stability,
        sensitivity=sensitivity,
        size_consistency=size_consistency,
        ansatz_disagreement=ansatz_disagreement,
        hysteresis=hysteresis,
    )
    assert stability.num_runs == 3
    assert 0.0 <= stability.converged_fraction <= 1.0
    assert stability.residual_max >= 0.0
    assert sensitivity.observable_shift_max >= 0.0
    assert size_consistency.observable_shift_max >= 0.0
    assert ansatz_disagreement.max_abs_gap >= 0.0
    assert hysteresis.observable_gap_max >= 0.0
    assert assessment.label in {"stable_classical", "fragile_classical", "frontier_or_uncertain"}
    assert assessment.score >= 0.0


def test_intrinsic_sample_contains_only_intrinsic_labeling() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    sample = build_intrinsic_sample(
        problem,
        sample_id=1,
        num_seeds=3,
        init_noise_scale=0.02,
        perturbation_scale=0.05,
    )
    assert sample["label_source"] == "intrinsic_only"
    assert "intrinsic_label" in sample
    assert "route_label" not in sample
    assert "size_consistency" in sample
    assert "ansatz_disagreement" in sample
    assert "hysteresis" in sample
    assert sample["problem_metadata"]["nsites"] == 4


def test_runtime_intrinsic_overlay_can_recover_or_escalate_routes() -> None:
    problem = ProblemSpec.tfim(Lx=2, Ly=2, J=1.0, h=0.8, g=0.0)
    report = analyze_runtime_intrinsic_corrmap(problem, num_seeds=2, init_noise_scale=0.01, perturbation_scale=0.02)
    recovered = apply_runtime_intrinsic_overlay(
        {
            "label": "uncertain",
            "recommended_action": "abstain_or_collect_stronger_evidence",
            "candidate_scores": {"mean_field": 0.7, "scalable_classical": 0.2},
            "abstained": True,
            "abstain_reason": "low_confidence",
        },
        report,
    )
    assert recovered["intrinsic_label"] in {"stable_classical", "fragile_classical", "frontier_or_uncertain"}
    assert recovered["label"] in {"mean_field", "scalable_classical", "quantum_frontier", "uncertain"}
    assert 0.0 <= recovered["mean_field_safety_score"] <= 1.0


def test_runtime_intrinsic_prior_promotes_strong_half_filled_hubbard_to_frontier() -> None:
    problem = ProblemSpec.hubbard(Lx=6, Ly=6, t=1.0, U=8.0, mu=4.0)
    report = analyze_runtime_intrinsic_corrmap(problem, num_seeds=2, init_noise_scale=0.01, perturbation_scale=0.02)
    assert "strong_coupling_half_filling" in report.assessment.reasons
    assert report.assessment.label in {"fragile_classical", "frontier_or_uncertain"}


def test_mean_field_safety_guard_can_block_mean_field_route() -> None:
    problem = ProblemSpec.tfim(Lx=6, Ly=6, J=1.0, h=0.9, g=0.1)
    report = analyze_runtime_intrinsic_corrmap(problem, num_seeds=2, init_noise_scale=0.01, perturbation_scale=0.02)
    guarded = apply_runtime_intrinsic_overlay(
        {
            "label": "mean_field",
            "recommended_action": "use_mean_field",
            "candidate_scores": {"mean_field": 0.9, "scalable_classical": 0.1},
            "abstained": False,
            "abstain_reason": None,
        },
        report,
    )
    assert guarded["label"] in {"mean_field", "scalable_classical", "quantum_frontier"}

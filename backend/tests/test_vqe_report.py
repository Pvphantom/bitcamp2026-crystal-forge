from app.analysis.vqe_report import build_vqe_report


def test_vqe_report_contains_safe_and_escalated_scenarios() -> None:
    report = build_vqe_report()

    assert report["summary"]["num_scenarios"] >= 2
    assert "tfim_easy_safe" in report["scenarios"]
    assert "tfim_quantum_escalation" in report["scenarios"]

    safe = report["scenarios"]["tfim_easy_safe"]
    escalated = report["scenarios"]["tfim_quantum_escalation"]

    assert safe["workflow_decision"]["measurement_mode"] == "not_needed"
    assert safe["qprobe"] is None
    assert escalated["workflow_decision"]["measurement_mode"] == "quantum_follow_on"
    assert escalated["workflow_decision"]["active_solver"] == "vqe"
    assert escalated["qprobe"] is not None
    assert escalated["solver_summary"]["strong_energy_gap"] is not None

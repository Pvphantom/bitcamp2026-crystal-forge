from __future__ import annotations

import json
from pathlib import Path

from app.analysis.vqe_report import build_vqe_report


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OUTPUT_PATH = ARTIFACT_DIR / "vqe_metrics.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_vqe_report()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    summary = report["summary"]
    print(
        "summary:"
        f" scenarios={summary['num_scenarios']}"
        f" escalated={summary['num_escalated']}"
        f" qprobe_runs={summary['num_qprobe_runs']}"
        f" avg_qprobe_savings={summary['avg_qprobe_savings']:.3f}"
    )
    for name, scenario in report["scenarios"].items():
        decision = scenario["workflow_decision"]
        solver_summary = scenario["solver_summary"]
        print(
            f"{name}: active={decision['active_solver']} "
            f"mode={decision['measurement_mode']} "
            f"cheap_gap={solver_summary['cheap_energy_gap']:.4f} "
            f"strong_gap={(solver_summary['strong_energy_gap'] if solver_summary['strong_energy_gap'] is not None else 'N/A')}"
        )


if __name__ == "__main__":
    main()

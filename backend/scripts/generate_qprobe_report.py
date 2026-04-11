from __future__ import annotations

import json
from pathlib import Path

from app.optimization.measurement_plan import search_minimal_measurement_plan
from app.physics.ed import ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.measurement_eval import NoiseModel


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OUTPUT_PATH = ARTIFACT_DIR / "qprobe_hero.json"


SCENARIOS = {
    "compression_win": {
        "params": {"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 4.0},
        "targets": ("D", "n", "Ms2", "Cs_max"),
        "tolerance": 0.03,
        "shots_per_group": 4000,
        "readout_flip_prob": 0.02,
    },
    "hard_observable": {
        "params": {"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0},
        "targets": ("D", "n", "Ms2", "K", "Cs_max"),
        "tolerance": 0.01,
        "shots_per_group": 2000,
        "readout_flip_prob": 0.08,
    },
}


def run_scenario(name: str, config: dict) -> dict:
    params = config["params"]
    h_op = build_hamiltonian(**params)
    _, state = ground_state(h_op)
    result = search_minimal_measurement_plan(
        Lx=params["Lx"],
        Ly=params["Ly"],
        t=params["t"],
        state=state,
        target_observables=config["targets"],
        tolerance=config["tolerance"],
        shots_per_group=config["shots_per_group"],
        noise_model=NoiseModel(readout_flip_prob=config["readout_flip_prob"]),
        seed=11,
    )
    return {
        "params": params,
        "targets": list(result.target_observables),
        "tolerance": result.tolerance,
        "full_cost": result.full_plan.cost,
        "recommended_cost": result.recommended_plan.cost,
        "measurement_savings": result.full_plan.cost - result.recommended_plan.cost,
        "success": result.success,
        "max_abs_error": result.max_abs_error,
        "message": result.message,
        "recommended_bases": result.recommended_plan.bases,
        "exact": result.exact,
        "estimated": result.estimated,
        "abs_error": result.abs_error,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = {name: run_scenario(name, config) for name, config in SCENARIOS.items()}
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    for name, payload in report.items():
        print(
            f"{name}: success={payload['success']} "
            f"full={payload['full_cost']} recommended={payload['recommended_cost']} "
            f"savings={payload['measurement_savings']} max_err={payload['max_abs_error']:.4f}"
        )


if __name__ == "__main__":
    main()

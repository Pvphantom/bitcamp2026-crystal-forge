from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR


LABELS = ("mean_field", "scalable_classical", "quantum_frontier")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=Path("backend/artifacts/regime_benchmark_refined.json"))
    parser.add_argument("--output-pt", type=Path, default=Path("backend/artifacts/regime_benchmark_refined.pt"))
    args = parser.parse_args()

    samples = build_regime_benchmark_refined()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(samples, indent=2))
    torch.save(samples, args.output_pt)
    print(json.dumps({"num_samples": len(samples), "json_path": str(args.output_json), "pt_path": str(args.output_pt)}, indent=2))


def build_regime_benchmark_refined() -> list[dict]:
    samples: list[dict] = []
    sample_id = 0
    for hard_label, probs, confidence, reason, problem in _benchmark_rows():
        sample_id += 1
        samples.append(
            {
                "sample_id": sample_id,
                "benchmark_label": hard_label,
                "benchmark_label_probs": probs,
                "benchmark_confidence": confidence,
                "label_source": "physics_prior_benchmark_refined",
                "rationale": reason,
                "problem": {
                    "model_family": problem.model_family,
                    "Lx": problem.Lx,
                    "Ly": problem.Ly,
                    "boundary": problem.lattice.boundary,
                    "parameters": {key: float(value) for key, value in problem.parameters.values.items()},
                },
            }
        )
    return samples


def _benchmark_rows() -> list[tuple[str, dict[str, float], str, str, ProblemSpec]]:
    rows: list[tuple[str, dict[str, float], str, str, ProblemSpec]] = []

    # High-confidence mean-field: clearly weak or strongly biased away from the interacting/critical region.
    for L in (6, 8):
        for U, mu in ((0.5, 0.0), (1.0, 2.0), (1.5, 4.0), (2.0, 4.0)):
            rows.append(
                (
                    "mean_field",
                    _probs(mean_field=0.85, scalable_classical=0.15, quantum_frontier=0.0),
                    "high",
                    "Clearly weak-coupling or strongly doped Hubbard point with high mean-field plausibility.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=mu),
                )
            )
    for L in (6, 8):
        for h, g in ((2.4, 0.5), (3.0, 0.5), (2.4, 0.0), (2.0, 0.5)):
            rows.append(
                (
                    "mean_field",
                    _probs(mean_field=0.85, scalable_classical=0.15, quantum_frontier=0.0),
                    "high",
                    "Clearly field-dominated TFIM point with suppressed fluctuation pressure.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    # Medium-confidence mean-field boundary points.
    for L in (6, 8):
        rows.append(
            (
                "mean_field",
                _probs(mean_field=0.6, scalable_classical=0.4, quantum_frontier=0.0),
                "medium",
                "Still expected to be mean-field-like, but close enough to the boundary that scalable classical methods remain plausible.",
                ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=1.0, mu=0.0),
            )
        )
        rows.append(
            (
                "mean_field",
                _probs(mean_field=0.6, scalable_classical=0.3, quantum_frontier=0.1),
                "medium",
                "Large-field TFIM point with low bias; mostly mean-field-like but no longer maximally clear-cut.",
                ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=2.0, g=0.0),
            )
        )

    # High-confidence scalable classical.
    for L in (6, 8):
        for U, mu in ((3.0, 4.0), (4.0, 2.0), (5.0, 2.0), (6.0, 4.0)):
            rows.append(
                (
                    "scalable_classical",
                    _probs(mean_field=0.1, scalable_classical=0.85, quantum_frontier=0.05),
                    "high",
                    "Intermediate-coupling Hubbard point beyond plain mean field but still plausibly classically tractable.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=mu),
                )
            )
    for L in (6, 8):
        for h, g in ((0.7, 0.5), (1.0, 0.5), (1.3, 0.5), (1.8, 0.5)):
            rows.append(
                (
                    "scalable_classical",
                    _probs(mean_field=0.05, scalable_classical=0.8, quantum_frontier=0.15),
                    "high",
                    "Biased TFIM point away from the clean critical line where scalable classical methods should be competitive.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    # Medium-confidence scalable classical boundary points.
    for L in (6, 8):
        rows.append(
            (
                "scalable_classical",
                _probs(mean_field=0.3, scalable_classical=0.6, quantum_frontier=0.1),
                "medium",
                "Intermediate Hubbard point close to the weak-coupling side of the boundary.",
                ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=3.0, mu=0.0),
            )
        )
        rows.append(
            (
                "scalable_classical",
                _probs(mean_field=0.05, scalable_classical=0.55, quantum_frontier=0.4),
                "medium",
                "TFIM point near the fluctuation-dominated boundary; scalable classical is favored but quantum pressure is substantial.",
                ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=2.2, g=1.0),
            )
        )

    # High-confidence quantum frontier.
    for L in (6, 8):
        for U in (8.0, 10.0, 12.0, 14.0):
            rows.append(
                (
                    "quantum_frontier",
                    _probs(mean_field=0.0, scalable_classical=0.1, quantum_frontier=0.9),
                    "high",
                    "Strongly correlated near-half-filled Hubbard point with pronounced frontier pressure.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=U / 2.0),
                )
            )
    for L in (6, 8):
        for h, g in ((0.8, 0.0), (1.0, 0.0), (1.2, 0.0), (1.1, 0.1)):
            rows.append(
                (
                    "quantum_frontier",
                    _probs(mean_field=0.0, scalable_classical=0.1, quantum_frontier=0.9),
                    "high",
                    "Near-critical low-bias TFIM point where collective quantum fluctuations should dominate.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    # Medium-confidence quantum frontier boundary points.
    for L in (6, 8):
        rows.append(
            (
                "quantum_frontier",
                _probs(mean_field=0.0, scalable_classical=0.3, quantum_frontier=0.7),
                "medium",
                "Lower-end strong-coupling half-filled Hubbard point with substantial but not maximal frontier pressure.",
                ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=6.0, mu=U / 2.0),
            )
        )
        rows.append(
            (
                "quantum_frontier",
                _probs(mean_field=0.0, scalable_classical=0.45, quantum_frontier=0.55),
                "medium",
                "TFIM point near the critical line but with a small longitudinal bias, making the frontier label plausible but not absolute.",
                ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=0.9, g=0.1),
            )
        )

    assert len(rows) == 60
    return rows


def _probs(*, mean_field: float, scalable_classical: float, quantum_frontier: float) -> dict[str, float]:
    total = mean_field + scalable_classical + quantum_frontier
    return {
        "mean_field": mean_field / total,
        "scalable_classical": scalable_classical / total,
        "quantum_frontier": quantum_frontier / total,
    }


if __name__ == "__main__":
    main()

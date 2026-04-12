from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("backend/artifacts/regime_benchmark.json"),
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        default=Path("backend/artifacts/regime_benchmark.pt"),
    )
    args = parser.parse_args()

    samples = build_regime_benchmark()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(samples, indent=2))
    torch.save(samples, args.output_pt)
    print(
        json.dumps(
            {
                "num_samples": len(samples),
                "label_counts": _label_counts(samples),
                "json_path": str(args.output_json),
                "pt_path": str(args.output_pt),
            },
            indent=2,
        )
    )


def build_regime_benchmark() -> list[dict]:
    samples: list[dict] = []
    sample_id = 0
    for label, reason, problem in _benchmark_problems():
        sample_id += 1
        samples.append(
            {
                "sample_id": sample_id,
                "benchmark_label": label,
                "label_source": "physics_prior_benchmark",
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


def _benchmark_problems() -> list[tuple[str, str, ProblemSpec]]:
    rows: list[tuple[str, str, ProblemSpec]] = []

    # Mean-field regime: weak coupling or strongly polarized limits where spatial correlations
    # are expected to be mild and uniform-order approximations are usually reliable.
    for L in (6, 8):
        for U, mu in ((0.5, 0.0), (1.0, 0.0), (1.0, 2.0), (1.5, 4.0), (2.0, 4.0)):
            rows.append(
                (
                    "mean_field",
                    "Weak-coupling or strongly doped Hubbard point where uniform mean field is a plausible baseline.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=mu),
                )
            )
    for L in (6, 8):
        for h, g in ((2.0, 0.0), (2.0, 0.5), (2.4, 0.0), (2.4, 0.5), (3.0, 0.5)):
            rows.append(
                (
                    "mean_field",
                    "Large-field TFIM point where polarization should suppress strong many-body structure.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    # Scalable classical regime: intermediate coupling or biased regimes that are often beyond
    # plain mean field, but still look amenable to scalable classical approximations.
    for L in (6, 8):
        for U, mu in ((3.0, 0.0), (3.0, 4.0), (4.0, 2.0), (5.0, 2.0), (6.0, 4.0)):
            rows.append(
                (
                    "scalable_classical",
                    "Intermediate-coupling Hubbard point where mean field can be fragile, but scalable classical methods remain plausible.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=mu),
                )
            )
    for L in (6, 8):
        for h, g in ((0.7, 0.5), (1.0, 0.5), (1.3, 0.5), (1.8, 0.5), (2.2, 1.0)):
            rows.append(
                (
                    "scalable_classical",
                    "Biased TFIM point away from the clean critical line where scalable classical structure should still be competitive.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    # Quantum-frontier regime: strong coupling near half filling or near-critical low-bias TFIM
    # where frustration/correlation effects are expected to dominate simple classical surrogates.
    for L in (6, 8):
        for U in (6.0, 8.0, 10.0, 12.0, 14.0):
            rows.append(
                (
                    "quantum_frontier",
                    "Strongly correlated near-half-filled Hubbard point intended to stress non-mean-field many-body structure.",
                    ProblemSpec.hubbard(Lx=L, Ly=L, t=1.0, U=U, mu=U / 2.0),
                )
            )
    for L in (6, 8):
        for h, g in ((0.8, 0.0), (1.0, 0.0), (1.2, 0.0), (0.9, 0.1), (1.1, 0.1)):
            rows.append(
                (
                    "quantum_frontier",
                    "Near-critical low-bias TFIM point where collective quantum fluctuations should be strongest.",
                    ProblemSpec.tfim(Lx=L, Ly=L, J=1.0, h=h, g=g),
                )
            )

    assert len(rows) == 60
    assert _label_counts_from_rows(rows) == {
        "mean_field": 20,
        "scalable_classical": 20,
        "quantum_frontier": 20,
    }
    return rows


def _label_counts(samples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        label = str(sample["benchmark_label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def _label_counts_from_rows(rows: list[tuple[str, str, ProblemSpec]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, _, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()

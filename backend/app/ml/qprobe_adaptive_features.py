from __future__ import annotations

from itertools import combinations

import torch

from app.domain.problem_spec import ProblemSpec
from app.ml.qprobe_operator_features import build_qprobe_operator_feature_vector
from app.physics.measurements import build_measurement_library_from_operator_map


def build_qprobe_adaptive_feature_vector(
    *,
    problem: ProblemSpec,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
) -> torch.Tensor:
    base = build_qprobe_operator_feature_vector(
        problem=problem,
        operator_map=operator_map,
        target_names=target_names,
        tolerance=tolerance,
        shots_per_group=shots_per_group,
        readout_flip_prob=readout_flip_prob,
    )
    adaptive = torch.tensor(
        _compression_redundancy_features(
            operator_map=operator_map,
            target_names=target_names,
        ),
        dtype=torch.float32,
    )
    return torch.cat([base, adaptive], dim=0)


def qprobe_adaptive_feature_dim() -> int:
    return 32


def _compression_redundancy_features(
    *,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
) -> list[float]:
    measurement_library = build_measurement_library_from_operator_map(operator_map)
    target_term_sets: list[set[str]] = []
    total_occurrences = 0
    term_counts: dict[str, int] = {}
    total_groups = 0
    merged_bases: set[str] = set()

    for name in target_names:
        op = operator_map[name].simplify()
        terms = {str(pauli) for pauli in op.paulis}
        target_term_sets.append(terms)
        total_occurrences += len(terms)
        for pauli in terms:
            term_counts[pauli] = term_counts.get(pauli, 0) + 1
        groups = measurement_library[name]
        total_groups += len(groups)
        for group in groups:
            merged_bases.add(group.basis)

    unique_terms = len(term_counts)
    duplicate_term_ratio = 0.0 if total_occurrences == 0 else 1.0 - unique_terms / total_occurrences
    shared_occurrences = sum(count for count in term_counts.values() if count > 1)
    shared_term_fraction = 0.0 if total_occurrences == 0 else shared_occurrences / total_occurrences
    basis_compression_ratio = 0.0 if total_groups == 0 else 1.0 - len(merged_bases) / total_groups

    pairwise_jaccards: list[float] = []
    for lhs, rhs in combinations(target_term_sets, 2):
        union = lhs | rhs
        pairwise_jaccards.append(0.0 if not union else len(lhs & rhs) / len(union))
    mean_pairwise_jaccard = 0.0 if not pairwise_jaccards else sum(pairwise_jaccards) / len(pairwise_jaccards)

    return [
        float(duplicate_term_ratio),
        float(shared_term_fraction),
        float(basis_compression_ratio),
        float(mean_pairwise_jaccard),
    ]

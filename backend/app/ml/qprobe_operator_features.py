from __future__ import annotations

from collections import Counter

import torch

from app.domain.problem_spec import ProblemSpec
from app.physics.measurements import MeasurementGroup, build_measurement_library_from_operator_map


def build_qprobe_operator_feature_vector(
    *,
    problem: ProblemSpec,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
    tolerance: float,
    shots_per_group: int,
    readout_flip_prob: float,
) -> torch.Tensor:
    measurement_library = build_measurement_library_from_operator_map(operator_map)
    selected_groups = [group for name in target_names for group in measurement_library[name]]
    merged_groups = _merged_groups_for_targets(measurement_library, target_names)

    family_flags = [
        1.0 if problem.model_family == "hubbard" else 0.0,
        1.0 if problem.model_family == "tfim" else 0.0,
    ]
    lattice = [float(problem.Lx), float(problem.Ly), float(problem.nsites)]
    params = _parameter_triplet(problem)
    target_stats = _target_stats(operator_map, target_names)
    group_stats = _group_stats(merged_groups)
    runtime = [float(tolerance), float(shots_per_group), float(readout_flip_prob)]
    return torch.tensor(
        [
            *family_flags,
            *lattice,
            *params,
            *target_stats,
            *group_stats,
            *runtime,
        ],
        dtype=torch.float32,
    )


def qprobe_operator_feature_dim() -> int:
    return 28


def _parameter_triplet(problem: ProblemSpec) -> list[float]:
    if problem.model_family == "hubbard":
        return [float(problem.t), float(problem.U), float(problem.mu)]
    return [float(problem.J), float(problem.h), float(problem.g)]


def _target_stats(operator_map: dict[str, object], target_names: tuple[str, ...]) -> list[float]:
    total_terms = 0
    total_abs_coeff = 0.0
    max_abs_coeff = 0.0
    diagonal_terms = 0
    support_sizes: list[int] = []
    unique_paulis: set[str] = set()
    for name in target_names:
        op = operator_map[name].simplify()
        total_terms += len(op)
        for pauli, coeff in zip(op.paulis, op.coeffs, strict=True):
            p = str(pauli)
            unique_paulis.add(p)
            abs_coeff = float(abs(complex(coeff)))
            total_abs_coeff += abs_coeff
            max_abs_coeff = max(max_abs_coeff, abs_coeff)
            support = sum(1 for symbol in p if symbol != "I")
            support_sizes.append(float(support))
            if set(p) <= {"I", "Z"}:
                diagonal_terms += 1
    mean_support = 0.0 if not support_sizes else sum(support_sizes) / len(support_sizes)
    max_support = 0.0 if not support_sizes else max(support_sizes)
    diagonal_fraction = 0.0 if total_terms == 0 else diagonal_terms / total_terms
    mean_abs_coeff = 0.0 if total_terms == 0 else total_abs_coeff / total_terms
    return [
        float(len(target_names)),
        float(total_terms),
        float(len(unique_paulis)),
        mean_support,
        max_support,
        diagonal_fraction,
        mean_abs_coeff,
        max_abs_coeff,
    ]


def _group_stats(groups: list[MeasurementGroup]) -> list[float]:
    basis_counter = Counter(_basis_family(group.basis) for group in groups)
    num_terms = [float(group.num_terms) for group in groups]
    support_sizes = [float(sum(1 for symbol in group.basis if symbol != "I")) for group in groups]
    return [
        float(len(groups)),
        0.0 if not num_terms else sum(num_terms) / len(num_terms),
        0.0 if not num_terms else max(num_terms),
        0.0 if not support_sizes else sum(support_sizes) / len(support_sizes),
        0.0 if not support_sizes else max(support_sizes),
        float(basis_counter.get("z", 0)),
        float(basis_counter.get("x", 0)),
        float(basis_counter.get("y", 0)),
        float(basis_counter.get("mixed", 0)),
    ]


def _basis_family(basis: str) -> str:
    active = {symbol for symbol in basis if symbol != "I"}
    if not active or active == {"Z"}:
        return "z"
    if active == {"X"}:
        return "x"
    if active == {"Y"}:
        return "y"
    return "mixed"


def _merged_groups_for_targets(
    measurement_library: dict[str, list[MeasurementGroup]],
    target_observables: tuple[str, ...],
) -> list[MeasurementGroup]:
    by_basis: dict[str, list[MeasurementGroup]] = {}
    for name in target_observables:
        for group in measurement_library[name]:
            by_basis.setdefault(group.basis, []).append(group)

    merged = []
    for basis, basis_groups in sorted(by_basis.items()):
        term_map: dict[str, complex] = {}
        for group in basis_groups:
            for term in group.terms:
                term_map[term.pauli] = term_map.get(term.pauli, 0.0j) + term.coeff
        merged.append(
            MeasurementGroup(
                name=f"merged:{basis}",
                basis=basis,
                terms=tuple(
                    type(basis_groups[0].terms[0])(pauli=pauli, coeff=coeff)
                    for pauli, coeff in sorted(term_map.items())
                    if abs(coeff) > 1e-12
                ),
            )
        )
    return merged

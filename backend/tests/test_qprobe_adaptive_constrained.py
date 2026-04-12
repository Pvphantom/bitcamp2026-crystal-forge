from __future__ import annotations

from app.analysis.qprobe_request_budget import (
    MAX_CUSTOM_BASIS_FAMILIES,
    MAX_CUSTOM_OPERATOR_SUPPORT,
    MAX_QPROBE_TARGETS_PER_REQUEST,
    MAX_TOTAL_PAULI_TERMS,
    validate_qprobe_request_budget,
)
from app.analysis.synthetic_operator_families import SYNTHETIC_OPERATOR_FAMILIES, build_synthetic_operator_bundle
from app.domain.problem_spec import ProblemSpec


def test_constrained_generator_candidates_can_be_budget_checked() -> None:
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0)
    seen_valid = False
    for family in SYNTHETIC_OPERATOR_FAMILIES:
        for num_targets in (1, 2, 3):
            bundle = build_synthetic_operator_bundle(problem=problem, family=family, num_targets=num_targets, seed=7)
            report = validate_qprobe_request_budget(
                target_names=bundle.target_names,
                operator_map=bundle.operator_map,
                has_custom_observables=True,
            )
            assert report.num_targets <= MAX_QPROBE_TARGETS_PER_REQUEST
            assert report.total_pauli_terms <= MAX_TOTAL_PAULI_TERMS
            assert report.max_operator_support <= MAX_CUSTOM_OPERATOR_SUPPORT
            assert len(report.basis_families) <= MAX_CUSTOM_BASIS_FAMILIES
            seen_valid = True
    assert seen_valid is True

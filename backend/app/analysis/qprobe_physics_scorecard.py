from __future__ import annotations

from collections import Counter

from app.optimization.measurement_plan import AdaptiveMeasurementStep


def build_qprobe_physics_scorecard(
    *,
    operator_map: dict[str, object],
    target_names: tuple[str, ...],
    step: AdaptiveMeasurementStep,
    readout_flip_prob: float,
) -> dict[str, float]:
    target_terms = {
        name: _term_weights(operator_map[name])
        for name in target_names
    }
    all_weights = Counter()
    for weights in target_terms.values():
        all_weights.update(weights)
    total_weight = float(sum(all_weights.values()))

    covered_paulis = {
        term.pauli
        for group in step.plan.groups
        for term in group.terms
    }
    covered_weight = sum(weight for pauli, weight in all_weights.items() if pauli in covered_paulis)
    uncovered_weight = max(0.0, total_weight - covered_weight)

    basis_weight = Counter()
    for pauli, weight in all_weights.items():
        basis_weight[_basis_family(pauli)] += weight

    chosen_basis_family = _basis_family(step.chosen_group.basis)
    unresolved_weight = 0.0
    unresolved_target_count = len(step.unresolved_targets)
    for name in step.unresolved_targets:
        unresolved_weight += float(sum(target_terms[name].values()))

    compatibility = 0.0 if total_weight == 0.0 else covered_weight / total_weight
    off_diagonal_weight = basis_weight["x"] + basis_weight["y"] + basis_weight["mixed"]
    off_diagonal_burden = 0.0 if total_weight == 0.0 else off_diagonal_weight / total_weight
    unresolved_mass_fraction = 0.0 if total_weight == 0.0 else unresolved_weight / total_weight
    uncertainty_pressure = float(step.max_uncertainty + readout_flip_prob)
    coefficient_concentration = 0.0
    if total_weight > 0.0:
        top_weights = sorted(all_weights.values(), reverse=True)[:3]
        coefficient_concentration = float(sum(top_weights) / total_weight)
    target_overlap = _mean_target_overlap(target_terms)

    return {
        "compatibility_score": compatibility,
        "uncovered_weight_fraction": 0.0 if total_weight == 0.0 else uncovered_weight / total_weight,
        "unresolved_mass_fraction": unresolved_mass_fraction,
        "off_diagonal_burden": off_diagonal_burden,
        "coefficient_concentration": coefficient_concentration,
        "target_overlap": target_overlap,
        "uncertainty_pressure": uncertainty_pressure,
        "chosen_basis_is_mixed": 1.0 if chosen_basis_family == "mixed" else 0.0,
    }


def _term_weights(operator) -> Counter[str]:
    simplified = operator.simplify()
    weights: Counter[str] = Counter()
    for pauli, coeff in zip(simplified.paulis, simplified.coeffs, strict=True):
        weights[str(pauli)] += float(abs(complex(coeff)))
    return weights


def _basis_family(pauli: str) -> str:
    active = {symbol for symbol in pauli if symbol != "I"}
    if not active or active == {"Z"}:
        return "z"
    if active == {"X"}:
        return "x"
    if active == {"Y"}:
        return "y"
    return "mixed"


def _mean_target_overlap(target_terms: dict[str, Counter[str]]) -> float:
    names = list(target_terms)
    if len(names) < 2:
        return 0.0
    overlaps: list[float] = []
    for i, lhs in enumerate(names):
        lhs_keys = set(target_terms[lhs])
        for rhs in names[i + 1 :]:
            rhs_keys = set(target_terms[rhs])
            union = lhs_keys | rhs_keys
            overlaps.append(0.0 if not union else len(lhs_keys & rhs_keys) / len(union))
    return float(sum(overlaps) / len(overlaps))

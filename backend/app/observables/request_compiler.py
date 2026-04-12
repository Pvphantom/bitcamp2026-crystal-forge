from __future__ import annotations

from typing import Iterable

from qiskit.quantum_info import SparsePauliOp

from app.domain.models import ObservableTargetSpecRequest
from app.domain.problem_spec import ProblemSpec
from app.observables.registry import ObservableRegistry


def resolve_observable_requests(
    *,
    problem: ProblemSpec,
    registry: ObservableRegistry,
    target_names: Iterable[str] | None = None,
    observable_specs: Iterable[ObservableTargetSpecRequest] | None = None,
    default_target_count: int = 3,
) -> tuple[tuple[str, ...], dict[str, SparsePauliOp]]:
    specs = list(observable_specs or [])
    if specs:
        operator_map: dict[str, SparsePauliOp] = {}
        for index, spec in enumerate(specs, start=1):
            target_name, operator = _compile_target_spec(
                problem=problem,
                registry=registry,
                spec=spec,
                fallback_name=f"custom_{index}",
            )
            if target_name in operator_map:
                raise ValueError(f"Duplicate observable target name: {target_name}")
            operator_map[target_name] = operator
        return tuple(operator_map.keys()), operator_map

    names = tuple(target_names or registry.names_for_family(problem.model_family)[:default_target_count])
    operator_map = {name: registry.operator(name, problem) for name in names}
    return names, operator_map


def _compile_target_spec(
    *,
    problem: ProblemSpec,
    registry: ObservableRegistry,
    spec: ObservableTargetSpecRequest,
    fallback_name: str,
) -> tuple[str, SparsePauliOp]:
    if spec.name is not None:
        target_name = spec.alias or spec.name
        return target_name, registry.operator(spec.name, problem)
    if not spec.pauli_terms:
        raise ValueError("Custom observable target must provide either a registered name or pauli_terms")
    target_name = spec.alias or fallback_name
    expected_qubits = _expected_operator_qubits(problem)
    terms: list[tuple[str, complex]] = []
    for term in spec.pauli_terms:
        pauli = term.pauli.strip()
        if len(pauli) != expected_qubits:
            raise ValueError(
                f"Pauli string length {len(pauli)} does not match expected operator qubit count {expected_qubits} for target {target_name}"
            )
        if any(symbol not in {"I", "X", "Y", "Z"} for symbol in pauli):
            raise ValueError(f"Unsupported Pauli string for target {target_name}: {pauli}")
        terms.append((pauli, complex(term.coeff_real, term.coeff_imag)))
    operator = SparsePauliOp.from_list(terms).simplify()
    if len(operator) == 0:
        raise ValueError(f"Observable target {target_name} simplified to the zero operator")
    return target_name, operator


def _expected_operator_qubits(problem: ProblemSpec) -> int:
    if problem.model_family == "tfim":
        return problem.nsites
    return problem.nqubits

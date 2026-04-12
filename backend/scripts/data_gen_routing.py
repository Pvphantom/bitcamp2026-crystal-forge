from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from app.analysis.routing_dataset import benchmark_sample_to_dict
from app.analysis.routing_label import RoutingLabelConfig, build_routing_label, route_family_for_solver
from app.analysis.solver_compare import compare_solver_results
from app.analysis.trust_features import build_trust_feature_groups, flatten_trust_feature_groups
from app.domain.problem_spec import ProblemSpec
from app.ml.schema import ARTIFACTS_DIR, DEFAULT_ROUTING_DATASET, RoutingBenchmarkSample, SolverBenchmarkOutcome
from app.solvers.base import SolverResult
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.registry import SolverRegistry
from app.solvers.tfim_mean_field import TFIMMeanFieldSolver
from app.solvers.vqe import VQESolver


GRIDS = {
    ("hubbard", 2, 2): {
        "U": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        "mu": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    },
    ("hubbard", 2, 3): {
        "U": [0.0, 2.0, 4.0, 8.0, 10.0],
        "mu": [0.0, 2.0, 4.0, 5.0],
    },
    ("tfim", 2, 2): {
        "J": [0.5, 1.0, 1.5],
        "h": [0.2, 0.6, 1.0, 1.5, 2.0],
        "g": [0.0, 0.5],
    },
    ("tfim", 2, 3): {
        "J": [1.0],
        "h": [0.4, 1.0, 1.8],
        "g": [0.0, 0.5],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ROUTING_DATASET)
    parser.add_argument("--reference-solver", type=str, default="exact_ed")
    parser.add_argument("--reference-quality", choices=["strong", "weak", "unknown"], default="strong")
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--observable-tolerance", type=float, default=0.08)
    parser.add_argument("--energy-tolerance", type=float, default=0.25)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    registry = _build_registry()
    policy = RoutingLabelConfig(
        observable_tolerance=args.observable_tolerance,
        energy_tolerance=args.energy_tolerance,
        allow_weak_labels=args.allow_weak_labels,
    )

    dataset: list[dict] = []
    sample_id = 0
    for (family, Lx, Ly), grid in GRIDS.items():
        for problem in _problems_for_grid(family, Lx, Ly, grid):
            if not registry.supports(args.reference_solver, problem):
                continue
            sample = build_routing_sample(
                problem=problem,
                registry=registry,
                reference_solver=args.reference_solver,
                reference_quality=args.reference_quality,
                policy=policy,
                sample_id=sample_id + 1,
            )
            dataset.append(benchmark_sample_to_dict(sample))
            sample_id += 1

    torch.save(dataset, args.output)
    print(f"saved {len(dataset)} routing benchmark samples to {args.output}")


def build_routing_sample(
    *,
    problem: ProblemSpec,
    registry: SolverRegistry,
    reference_solver: str,
    reference_quality: str,
    policy: RoutingLabelConfig,
    sample_id: int,
) -> RoutingBenchmarkSample:
    available = registry.available_for(problem)
    candidate_solvers = [name for name in available if name in _candidate_solver_names(problem)]
    reference_result, reference_runtime = _solve_with_timing(registry, reference_solver, problem)
    cheap_solver_name = "mean_field" if problem.model_family == "hubbard" else "tfim_mean_field"
    cheap_result, cheap_runtime = _solve_with_timing(registry, cheap_solver_name, problem)

    outcomes: dict[str, SolverBenchmarkOutcome] = {
        reference_solver: _reference_outcome(reference_solver, reference_result, reference_runtime),
    }
    if cheap_solver_name != reference_solver:
        outcomes[cheap_solver_name] = _candidate_outcome(
            cheap_solver_name,
            cheap_result,
            reference=reference_result,
            runtime_s=cheap_runtime,
            problem=problem,
        )

    for solver_name in candidate_solvers:
        if solver_name in outcomes:
            continue
        result, runtime_s = _solve_with_timing(registry, solver_name, problem)
        outcomes[solver_name] = _candidate_outcome(
            solver_name,
            result,
            reference=reference_result,
            runtime_s=runtime_s,
            problem=problem,
        )

    feature_groups = build_trust_feature_groups(problem, cheap_result)
    features = flatten_trust_feature_groups(feature_groups)
    label = build_routing_label(
        outcomes,
        reference_solver=reference_solver,
        reference_quality=reference_quality,
        config=policy,
    )
    metadata = _problem_metadata(problem)
    metadata.update(
        {
            "sample_id": sample_id,
            "cheap_solver": cheap_solver_name,
            "reference_solver": reference_solver,
            "available_solvers": candidate_solvers,
        }
    )
    notes = []
    if reference_quality != "strong":
        notes.append("Reference quality is weaker than exact-oracle supervision.")
    if label.route_label == "uncertain":
        notes.append("Routing label abstained because the benchmark evidence was insufficient or policy blocked weak labels.")

    return RoutingBenchmarkSample(
        features=features,
        feature_groups=feature_groups,
        route_label=label.route_label,
        problem_metadata=metadata,
        solver_outcomes=outcomes,
        reference_solver=reference_solver,
        reference_quality=reference_quality,
        label_source=label.label_source,
        notes=notes,
    )


def _build_registry() -> SolverRegistry:
    registry = SolverRegistry()
    registry.register(ExactEDSolver())
    registry.register(MeanFieldSolver())
    registry.register(TFIMMeanFieldSolver())
    registry.register(VQESolver())
    return registry


def _candidate_solver_names(problem: ProblemSpec) -> list[str]:
    if problem.model_family == "hubbard":
        return ["mean_field", "exact_ed"]
    if problem.model_family == "tfim":
        return ["tfim_mean_field", "exact_ed", "vqe"]
    raise ValueError(f"Unsupported model family: {problem.model_family}")


def _reference_outcome(solver_name: str, result: SolverResult, runtime_s: float) -> SolverBenchmarkOutcome:
    return SolverBenchmarkOutcome(
        solver_name=solver_name,
        family=route_family_for_solver(solver_name),
        succeeded=True,
        runtime_s=runtime_s,
        observables={key: float(value) for key, value in result.global_observables.items()},
        max_abs_error=0.0,
        energy=float(result.energy),
        energy_error=0.0,
        cost_class=_default_cost_class_for_solver(solver_name),
        notes=["reference_solver"],
    )


def _candidate_outcome(
    solver_name: str,
    result: SolverResult,
    *,
    reference: SolverResult,
    runtime_s: float,
    problem: ProblemSpec,
) -> SolverBenchmarkOutcome:
    comparison = compare_solver_results(problem, reference, result)
    return SolverBenchmarkOutcome(
        solver_name=solver_name,
        family=route_family_for_solver(solver_name),
        succeeded=True,
        runtime_s=runtime_s,
        observables={key: float(value) for key, value in result.global_observables.items()},
        abs_error=dict(comparison.abs_error),
        rel_error=dict(comparison.rel_error),
        max_abs_error=float(comparison.max_abs_error),
        energy=float(result.energy),
        energy_error=float(comparison.energy_error),
        cost_class=_default_cost_class_for_solver(solver_name),
        notes=[],
    )


def _default_cost_class_for_solver(solver_name: str) -> str:
    route_family = route_family_for_solver(solver_name)
    if route_family == "mean_field":
        return "cheap"
    if route_family in {"scalable_classical", "oracle_reference"}:
        return "expensive"
    return "frontier"


def _solve_with_timing(registry: SolverRegistry, solver_name: str, problem: ProblemSpec) -> tuple[SolverResult, float]:
    start = perf_counter()
    result = registry.get(solver_name).solve(problem)
    runtime_s = perf_counter() - start
    return result, runtime_s


def _problem_metadata(problem: ProblemSpec) -> dict[str, object]:
    metadata: dict[str, object] = {
        "family": problem.model_family,
        "Lx": problem.Lx,
        "Ly": problem.Ly,
        "nsites": problem.nsites,
        "boundary": problem.lattice.boundary,
    }
    metadata.update({key: float(value) for key, value in problem.parameters.values.items()})
    return metadata


def _problems_for_grid(family: str, Lx: int, Ly: int, grid: dict[str, list[float]]) -> list[ProblemSpec]:
    problems: list[ProblemSpec] = []
    if family == "hubbard":
        for U in grid["U"]:
            for mu in grid["mu"]:
                problems.append(ProblemSpec.hubbard(Lx=Lx, Ly=Ly, t=1.0, U=U, mu=mu))
        return problems
    if family == "tfim":
        for J in grid["J"]:
            for h in grid["h"]:
                for g in grid["g"]:
                    problems.append(ProblemSpec.tfim(Lx=Lx, Ly=Ly, J=J, h=h, g=g))
        return problems
    raise ValueError(f"Unsupported model family: {family}")


if __name__ == "__main__":
    main()

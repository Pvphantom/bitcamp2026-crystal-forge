from __future__ import annotations

from app.domain.problem_spec import ProblemSpec
from app.observables.registry import ObservableRegistry, build_default_observable_registry
from app.physics.ed import expectation_value, ground_state
from app.physics.hamiltonian import build_hamiltonian
from app.physics.observables import extract_site_observables_from_statevector
from app.physics.tfim import (
    build_tfim_hamiltonian,
    build_tfim_site_x_operators,
    build_tfim_site_z_operators,
)
from app.solvers.base import BaseSolver, SolverResult


class ExactEDSolver(BaseSolver):
    name = "exact_ed"

    def __init__(self, observable_registry: ObservableRegistry | None = None) -> None:
        self.observable_registry = observable_registry or build_default_observable_registry()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family in {"hubbard", "tfim"}

    def solve(self, problem: ProblemSpec) -> SolverResult:
        if problem.model_family == "hubbard":
            h_op = build_hamiltonian(problem.Lx, problem.Ly, problem.t, problem.U, problem.mu)
        elif problem.model_family == "tfim":
            h_op = build_tfim_hamiltonian(problem.Lx, problem.Ly, problem.J, problem.h, problem.g)
        else:
            raise ValueError(f"Unsupported model family for exact solver: {problem.model_family}")
        energy, state = ground_state(h_op)
        observable_ops = self.observable_registry.operator_map(problem)
        global_observables = {
            name: expectation_value(op, state)
            for name, op in observable_ops.items()
        }
        global_observables["energy"] = energy
        if problem.model_family == "hubbard":
            site_observables = extract_site_observables_from_statevector(problem.Lx, problem.Ly, state)
        else:
            mz_ops = build_tfim_site_z_operators(problem.Lx, problem.Ly)
            mx_ops = build_tfim_site_x_operators(problem.Lx, problem.Ly)
            site_observables = {
                "Mz_site": [expectation_value(op, state) for op in mz_ops],
                "Mx_site": [expectation_value(op, state) for op in mx_ops],
            }
        bond_observables = {
            bond: expectation_value(op, state)
            for bond, op in self.observable_registry.bond_operators(problem).items()
        }
        return SolverResult(
            solver_name=self.name,
            energy=energy,
            global_observables=global_observables,
            site_observables=site_observables,
            bond_observables=bond_observables,
            statevector=state,
            metadata={"method": "exact_diagonalization"},
        )

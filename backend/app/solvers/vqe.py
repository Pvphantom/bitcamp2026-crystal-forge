from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from app.domain.problem_spec import ProblemSpec
from app.observables.registry import ObservableRegistry, build_default_observable_registry
from app.physics.ed import expectation_value
from app.physics.lattice import nn_bonds
from app.physics.tfim import build_tfim_site_x_operators, build_tfim_site_z_operators
from app.solvers.base import BaseSolver, SolverResult


@dataclass(frozen=True)
class VQESettings:
    depth: int = 3
    maxiter: int = 200
    tol: float = 1e-6
    optimizer: str = "COBYLA"
    init_scale: float = 0.05
    seed: int = 7


class VQESolver(BaseSolver):
    """Problem-inspired TFIM VQE on small qubit lattices.

    The ansatz starts from |+...+> to reflect the transverse-field limit and applies
    repeated layers of:
      1. nearest-neighbor ZZ entanglers,
      2. uniform RY rotations,
      3. uniform RZ rotations.
    """

    name = "vqe"

    def __init__(
        self,
        settings: VQESettings | None = None,
        observable_registry: ObservableRegistry | None = None,
    ) -> None:
        self.settings = settings or VQESettings()
        self.observable_registry = observable_registry or build_default_observable_registry()

    def supports(self, problem: ProblemSpec) -> bool:
        return problem.model_family == "tfim"

    def solve(self, problem: ProblemSpec) -> SolverResult:
        if not self.supports(problem):
            raise ValueError(f"VQE solver does not support model family: {problem.model_family}")

        observable_ops = self.observable_registry.operator_map(problem)
        energy_op = self._hamiltonian(problem)
        bonds = nn_bonds(problem.Lx, problem.Ly)
        rng = np.random.default_rng(self.settings.seed)
        x0 = rng.normal(scale=self.settings.init_scale, size=3 * self.settings.depth)

        history: list[float] = []
        best_energy = float("inf")
        best_state: np.ndarray | None = None

        def objective(theta: np.ndarray) -> float:
            nonlocal best_energy, best_state
            state = self._statevector(problem, theta, bonds)
            energy = expectation_value(energy_op, state)
            history.append(float(energy))
            if energy < best_energy:
                best_energy = float(energy)
                best_state = state.copy()
            return float(energy)

        result = minimize(
            objective,
            x0=x0,
            method=self.settings.optimizer,
            tol=self.settings.tol,
            options={"maxiter": self.settings.maxiter},
        )

        final_theta = np.asarray(result.x, dtype=float)
        final_state = best_state if best_state is not None else self._statevector(problem, final_theta, bonds)
        final_energy = expectation_value(energy_op, final_state)
        global_observables = {
            name: expectation_value(op, final_state)
            for name, op in observable_ops.items()
        }
        global_observables["energy"] = final_energy

        mz_ops = build_tfim_site_z_operators(problem.Lx, problem.Ly)
        mx_ops = build_tfim_site_x_operators(problem.Lx, problem.Ly)
        site_observables = {
            "Mz_site": [expectation_value(op, final_state) for op in mz_ops],
            "Mx_site": [expectation_value(op, final_state) for op in mx_ops],
        }
        bond_observables = {
            bond: expectation_value(op, final_state)
            for bond, op in self.observable_registry.bond_operators(problem).items()
        }

        return SolverResult(
            solver_name=self.name,
            energy=float(final_energy),
            global_observables=global_observables,
            site_observables=site_observables,
            bond_observables=bond_observables,
            statevector=final_state,
            metadata={
                "method": "tfim_vqe",
                "depth": self.settings.depth,
                "parameter_count": int(final_theta.size),
                "iterations": int(getattr(result, "nit", len(history))),
                "evaluations": int(getattr(result, "nfev", len(history))),
                "converged": bool(result.success),
                "optimizer": self.settings.optimizer,
                "final_parameters": final_theta.tolist(),
                "energy_history": history,
            },
        )

    @staticmethod
    def _hamiltonian(problem: ProblemSpec):
        from app.physics.tfim import build_tfim_hamiltonian

        return build_tfim_hamiltonian(problem.Lx, problem.Ly, problem.J, problem.h, problem.g)

    def _statevector(
        self,
        problem: ProblemSpec,
        theta: np.ndarray,
        bonds: list[tuple[int, int]],
    ) -> np.ndarray:
        nsites = problem.nsites
        circuit = QuantumCircuit(nsites)
        circuit.h(range(nsites))
        params = np.asarray(theta, dtype=float).reshape(self.settings.depth, 3)
        for gamma, beta, alpha in params:
            for i, j in bonds:
                circuit.rzz(float(gamma), i, j)
            for qubit in range(nsites):
                circuit.ry(float(beta), qubit)
                circuit.rz(float(alpha), qubit)
        return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)

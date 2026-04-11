from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.problem_spec import ProblemSpec
from app.solvers.base import BaseSolver


@dataclass
class SolverRegistry:
    _solvers: dict[str, BaseSolver] = field(default_factory=dict)

    def register(self, solver: BaseSolver) -> None:
        self._solvers[solver.name] = solver

    def get(self, name: str) -> BaseSolver:
        return self._solvers[name]

    def supports(self, name: str, problem: ProblemSpec) -> bool:
        return self.get(name).supports(problem)

    def available_for(self, problem: ProblemSpec) -> list[str]:
        return [name for name, solver in self._solvers.items() if solver.supports(problem)]


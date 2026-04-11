from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.domain.problem_spec import ProblemSpec


@dataclass(frozen=True)
class SolverResult:
    solver_name: str
    energy: float
    global_observables: dict[str, float]
    site_observables: dict[str, list[float]]
    bond_observables: dict[tuple[int, int], float]
    statevector: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class BaseSolver(ABC):
    name: str

    @abstractmethod
    def supports(self, problem: ProblemSpec) -> bool:
        raise NotImplementedError

    @abstractmethod
    def solve(self, problem: ProblemSpec) -> SolverResult:
        raise NotImplementedError


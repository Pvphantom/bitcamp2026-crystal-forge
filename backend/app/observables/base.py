from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qiskit.quantum_info import SparsePauliOp

from app.domain.problem_spec import ProblemSpec


ObservableBuilder = Callable[[ProblemSpec], SparsePauliOp]


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    label: str
    description: str
    families: tuple[str, ...]
    builder: ObservableBuilder


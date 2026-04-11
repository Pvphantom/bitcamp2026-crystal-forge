from __future__ import annotations

from dataclasses import dataclass, field

from qiskit.quantum_info import SparsePauliOp

from app.domain.problem_spec import ProblemSpec
from app.observables.base import ObservableSpec
from app.physics.observables import (
    build_bond_spin_correlator_operators,
    build_double_occ,
    build_filling,
    build_kinetic,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
)


def _build_double_occ(problem: ProblemSpec) -> SparsePauliOp:
    return build_double_occ(problem.Lx, problem.Ly)


def _build_filling(problem: ProblemSpec) -> SparsePauliOp:
    return build_filling(problem.Lx, problem.Ly)


def _build_ms2(problem: ProblemSpec) -> SparsePauliOp:
    return build_staggered_magnetization_squared(problem.Lx, problem.Ly)


def _build_kinetic(problem: ProblemSpec) -> SparsePauliOp:
    return build_kinetic(problem.Lx, problem.Ly, problem.t)


def _build_cs_max(problem: ProblemSpec) -> SparsePauliOp:
    return build_spin_correlator_maxdist(problem.Lx, problem.Ly)


@dataclass
class ObservableRegistry:
    _observables: dict[str, ObservableSpec] = field(default_factory=dict)

    def register(self, spec: ObservableSpec) -> None:
        self._observables[spec.name] = spec

    def names_for_family(self, family: str) -> list[str]:
        return [name for name, spec in self._observables.items() if family in spec.families]

    def operator(self, name: str, problem: ProblemSpec) -> SparsePauliOp:
        return self._observables[name].builder(problem)

    def operator_map(self, problem: ProblemSpec) -> dict[str, SparsePauliOp]:
        return {
            name: self.operator(name, problem)
            for name in self.names_for_family(problem.model_family)
        }

    def hubbard_bond_operators(self, problem: ProblemSpec) -> dict[tuple[int, int], SparsePauliOp]:
        return build_bond_spin_correlator_operators(problem.Lx, problem.Ly)


def build_default_observable_registry() -> ObservableRegistry:
    registry = ObservableRegistry()
    registry.register(
        ObservableSpec(
            name="D",
            label="Double occupancy",
            description="How often two electrons occupy the same site.",
            families=("hubbard",),
            builder=_build_double_occ,
        )
    )
    registry.register(
        ObservableSpec(
            name="n",
            label="Average filling",
            description="Average charge per site.",
            families=("hubbard",),
            builder=_build_filling,
        )
    )
    registry.register(
        ObservableSpec(
            name="Ms2",
            label="Spin alternation strength",
            description="Antiferromagnetic checkerboard strength.",
            families=("hubbard",),
            builder=_build_ms2,
        )
    )
    registry.register(
        ObservableSpec(
            name="K",
            label="Motion / kinetic signal",
            description="Average hopping contribution per bond.",
            families=("hubbard",),
            builder=_build_kinetic,
        )
    )
    registry.register(
        ObservableSpec(
            name="Cs_max",
            label="Long-range spin link",
            description="Most distant spin-spin correlation on the cluster.",
            families=("hubbard",),
            builder=_build_cs_max,
        )
    )
    return registry


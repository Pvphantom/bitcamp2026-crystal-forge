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
    build_singlet_pair_density,
    build_singlet_pair_span,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
)
from app.physics.tfim import (
    build_tfim_bond_zz_operators,
    build_tfim_mx,
    build_tfim_mz,
    build_tfim_staggered_mz2,
    build_tfim_z_span,
    build_tfim_zz_nn,
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


def _build_pair_nn(problem: ProblemSpec) -> SparsePauliOp:
    return build_singlet_pair_density(problem.Lx, problem.Ly)


def _build_pair_span(problem: ProblemSpec) -> SparsePauliOp:
    return build_singlet_pair_span(problem.Lx, problem.Ly)


def _build_tfim_mz(problem: ProblemSpec) -> SparsePauliOp:
    return build_tfim_mz(problem.Lx, problem.Ly)


def _build_tfim_mx(problem: ProblemSpec) -> SparsePauliOp:
    return build_tfim_mx(problem.Lx, problem.Ly)


def _build_tfim_zz_nn(problem: ProblemSpec) -> SparsePauliOp:
    return build_tfim_zz_nn(problem.Lx, problem.Ly)


def _build_tfim_staggered(problem: ProblemSpec) -> SparsePauliOp:
    return build_tfim_staggered_mz2(problem.Lx, problem.Ly)


def _build_tfim_z_span(problem: ProblemSpec) -> SparsePauliOp:
    return build_tfim_z_span(problem.Lx, problem.Ly)


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

    def bond_operators(self, problem: ProblemSpec) -> dict[tuple[int, int], SparsePauliOp]:
        if problem.model_family == "hubbard":
            return build_bond_spin_correlator_operators(problem.Lx, problem.Ly)
        if problem.model_family == "tfim":
            return build_tfim_bond_zz_operators(problem.Lx, problem.Ly)
        return {}


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
    registry.register(
        ObservableSpec(
            name="Pair_nn",
            label="Nearest-neighbor singlet pairing",
            description="Average nearest-neighbor singlet pair density.",
            families=("hubbard",),
            builder=_build_pair_nn,
        )
    )
    registry.register(
        ObservableSpec(
            name="Pair_span",
            label="Long-range pair coherence",
            description="Longest-distance singlet-pair coherence on the cluster.",
            families=("hubbard",),
            builder=_build_pair_span,
        )
    )
    registry.register(
        ObservableSpec(
            name="Mz",
            label="Average Z magnetization",
            description="Average longitudinal spin polarization.",
            families=("tfim",),
            builder=_build_tfim_mz,
        )
    )
    registry.register(
        ObservableSpec(
            name="Mx",
            label="Average X magnetization",
            description="Average transverse-field alignment.",
            families=("tfim",),
            builder=_build_tfim_mx,
        )
    )
    registry.register(
        ObservableSpec(
            name="ZZ_nn",
            label="Nearest-neighbor ZZ order",
            description="Average nearest-neighbor Ising correlation.",
            families=("tfim",),
            builder=_build_tfim_zz_nn,
        )
    )
    registry.register(
        ObservableSpec(
            name="Mstag2",
            label="Staggered Z order strength",
            description="Squared staggered longitudinal magnetization.",
            families=("tfim",),
            builder=_build_tfim_staggered,
        )
    )
    registry.register(
        ObservableSpec(
            name="Z_span",
            label="Long-range Z link",
            description="Longest-distance longitudinal spin correlation on the cluster.",
            families=("tfim",),
            builder=_build_tfim_z_span,
        )
    )
    return registry

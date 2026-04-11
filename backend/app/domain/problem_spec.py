from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatticeSpec:
    Lx: int
    Ly: int
    boundary: str = "open"

    @property
    def nsites(self) -> int:
        return self.Lx * self.Ly

    @property
    def nqubits(self) -> int:
        return 2 * self.nsites


@dataclass(frozen=True)
class ParameterSpec:
    values: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.values.get(key, default)


@dataclass(frozen=True)
class ProblemSpec:
    model_family: str
    lattice: LatticeSpec
    parameters: ParameterSpec
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def hubbard(
        cls,
        *,
        Lx: int,
        Ly: int,
        t: float,
        U: float,
        mu: float,
        boundary: str = "open",
        metadata: dict[str, object] | None = None,
    ) -> "ProblemSpec":
        return cls(
            model_family="hubbard",
            lattice=LatticeSpec(Lx=Lx, Ly=Ly, boundary=boundary),
            parameters=ParameterSpec(values={"t": t, "U": U, "mu": mu}),
            metadata=metadata or {},
        )

    @property
    def Lx(self) -> int:
        return self.lattice.Lx

    @property
    def Ly(self) -> int:
        return self.lattice.Ly

    @property
    def nsites(self) -> int:
        return self.lattice.nsites

    @property
    def nqubits(self) -> int:
        return self.lattice.nqubits

    @property
    def t(self) -> float:
        return float(self.parameters.values["t"])

    @property
    def U(self) -> float:
        return float(self.parameters.values["U"])

    @property
    def mu(self) -> float:
        return float(self.parameters.values["mu"])


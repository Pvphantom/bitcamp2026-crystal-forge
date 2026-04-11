from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from app.domain.models import (
    BondSnapshot,
    CreateStateRequest,
    EvolveRequest,
    ExportStateResponse,
    LatticeSnapshot,
    MetricsSummaryResponse,
    ObservablesResponse,
    PhasePredictionResponse,
    PlaceConfigurationRequest,
    SetParamsRequest,
    SiteSnapshot,
)
from app.ml.infer import MetricsReader, PhaseInferenceEngine
from app.ml.schema import build_graph_sample, classify_phase_rule
from app.physics.ed import expectation_value, ground_state, operator_matrix
from app.physics.hamiltonian import build_hamiltonian
from app.physics.lattice import nn_bonds
from app.physics.observables import (
    build_bond_spin_correlator_operators,
    build_double_occ,
    build_filling,
    build_kinetic,
    build_spin_correlator_maxdist,
    build_staggered_magnetization_squared,
    extract_site_observables_from_statevector,
)
from app.physics.state_prep import (
    basis_statevector_from_occupations,
    prepare_product_state_circuit,
)


@dataclass
class StateConfig:
    Lx: int = 2
    Ly: int = 2
    t: float = 1.0
    U: float = 4.0
    mu: float = 2.0


class HubbardGameStateService:
    """Stateful backend service for small-lattice Hubbard simulation."""

    def __init__(self) -> None:
        self.config = StateConfig()
        self.statevector: np.ndarray | None = None
        self.h_op = None
        self.h_matrix: np.ndarray | None = None
        self.observable_ops: dict[str, object] = {}
        self.phase_inference = PhaseInferenceEngine()
        self.metrics_reader = MetricsReader()
        self.reset(default_state="neel")

    def create_state(self, payload: CreateStateRequest) -> ExportStateResponse:
        self.config = StateConfig(**payload.model_dump())
        self.reset(default_state="neel")
        return self.export_state()

    def set_params(self, payload: SetParamsRequest) -> ExportStateResponse:
        updates = payload.model_dump(exclude_none=True)
        self.config = StateConfig(**{**self.config.__dict__, **updates})
        self._rebuild_hamiltonian_and_observables()
        return self.export_state()

    def reset(self, *, default_state: str = "neel") -> ExportStateResponse:
        self._rebuild_hamiltonian_and_observables()
        self.statevector = basis_statevector_from_occupations(
            self.config.Lx,
            self.config.Ly,
            default=default_state,
        )
        return self.export_state()

    def reset_to_neel(self) -> ExportStateResponse:
        return self.reset(default_state="neel")

    def place_configuration(self, payload: PlaceConfigurationRequest) -> ExportStateResponse:
        override_map = {
            (entry.x, entry.y, entry.spin): entry.occupied
            for entry in payload.occupations
        }
        self.statevector = basis_statevector_from_occupations(
            self.config.Lx,
            self.config.Ly,
            occupations=override_map,
            default=payload.default_state,
        )
        return self.export_state()

    def evolve(self, payload: EvolveRequest) -> ExportStateResponse:
        self._ensure_state()
        assert self.h_matrix is not None
        unitary = expm(-1j * self.h_matrix * payload.dt)
        for _ in range(payload.steps):
            self.statevector = unitary @ self.statevector
        norm = np.linalg.norm(self.statevector)
        if norm > 0:
            self.statevector = self.statevector / norm
        return self.export_state()

    def set_ground_state(self) -> ExportStateResponse:
        self._ensure_state()
        assert self.h_op is not None
        _, psi0 = ground_state(self.h_op)
        self.statevector = psi0
        return self.export_state()

    def get_observables(self) -> ObservablesResponse:
        self._ensure_state()
        return self._build_observables()

    def predict_phase(self) -> PhasePredictionResponse:
        obs = self.get_observables()
        sample = self._build_ml_sample()
        inferred = self.phase_inference.predict(sample)
        if inferred is not None:
            return PhasePredictionResponse(**inferred)

        probabilities = {
            "Metal": 0.1,
            "Mott Insulator": 0.1,
            "Antiferromagnet": 0.1,
            "Singlet-rich": 0.1,
        }

        label = classify_phase_rule(self.config.U, obs.n, obs.Ms2)
        if label == "Metal":
            probabilities["Metal"] = 0.7
        elif label == "Mott Insulator":
            probabilities["Mott Insulator"] = 0.75
        elif label == "Antiferromagnet":
            probabilities["Antiferromagnet"] = 0.8
        else:
            probabilities["Singlet-rich"] = 0.6

        total = sum(probabilities.values())
        probabilities = {key: value / total for key, value in probabilities.items()}
        confidence = max(probabilities.values())
        return PhasePredictionResponse(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            model_status=self.phase_inference.status(source="fallback-rules"),
        )

    def get_metrics(self) -> MetricsSummaryResponse:
        return MetricsSummaryResponse(**self.metrics_reader.summary())

    def export_state(self) -> ExportStateResponse:
        self._ensure_state()
        return ExportStateResponse(
            lattice=self._build_lattice_snapshot(),
            observables=self._build_observables(),
            phase=self.predict_phase(),
            metrics=self.get_metrics(),
        )

    def get_state_circuit(self):
        return prepare_product_state_circuit(self.config.Lx, self.config.Ly, default="neel")

    def _ensure_state(self) -> None:
        if self.statevector is None:
            self.statevector = basis_statevector_from_occupations(
                self.config.Lx,
                self.config.Ly,
                default="neel",
            )

    def _rebuild_hamiltonian_and_observables(self) -> None:
        self.h_op = build_hamiltonian(
            self.config.Lx,
            self.config.Ly,
            self.config.t,
            self.config.U,
            self.config.mu,
        )
        self.h_matrix = operator_matrix(self.h_op)
        self.observable_ops = {
            "D": build_double_occ(self.config.Lx, self.config.Ly),
            "n": build_filling(self.config.Lx, self.config.Ly),
            "Ms2": build_staggered_magnetization_squared(self.config.Lx, self.config.Ly),
            "K": build_kinetic(self.config.Lx, self.config.Ly, self.config.t),
            "Cs_max": build_spin_correlator_maxdist(self.config.Lx, self.config.Ly),
            "bond_ops": build_bond_spin_correlator_operators(self.config.Lx, self.config.Ly),
        }

    def _build_observables(self) -> ObservablesResponse:
        self._ensure_state()
        assert self.h_op is not None
        assert self.statevector is not None

        return ObservablesResponse(
            D=expectation_value(self.observable_ops["D"], self.statevector),
            n=expectation_value(self.observable_ops["n"], self.statevector),
            Ms2=expectation_value(self.observable_ops["Ms2"], self.statevector),
            K=expectation_value(self.observable_ops["K"], self.statevector),
            Cs_max=expectation_value(self.observable_ops["Cs_max"], self.statevector),
            energy=expectation_value(self.h_op, self.statevector),
        )

    def _build_lattice_snapshot(self) -> LatticeSnapshot:
        self._ensure_state()
        assert self.statevector is not None
        site_data = extract_site_observables_from_statevector(
            self.config.Lx,
            self.config.Ly,
            self.statevector,
        )

        sites: list[SiteSnapshot] = []
        for y in range(self.config.Ly):
            for x in range(self.config.Lx):
                i = x + self.config.Lx * y
                sites.append(
                    SiteSnapshot(
                        i=i,
                        x=x,
                        y=y,
                        n_up=site_data["n_up"][i],
                        n_dn=site_data["n_dn"][i],
                        double_occ=site_data["D_site"][i],
                        sz=site_data["Sz_site"][i],
                    )
                )

        bonds: list[BondSnapshot] = []
        for i, j in nn_bonds(self.config.Lx, self.config.Ly):
            strength = expectation_value(self.observable_ops["bond_ops"][(i, j)], self.statevector)
            bonds.append(BondSnapshot(i=i, j=j, strength=strength))

        return LatticeSnapshot(
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            sites=sites,
            bonds=bonds,
        )

    def _build_ml_sample(self) -> dict[str, object]:
        assert self.statevector is not None
        site_data = extract_site_observables_from_statevector(
            self.config.Lx,
            self.config.Ly,
            self.statevector,
        )
        bond_strengths = {
            bond: expectation_value(op, self.statevector)
            for bond, op in self.observable_ops["bond_ops"].items()
        }
        node_features = []
        for y in range(self.config.Ly):
            for x in range(self.config.Lx):
                i = x + self.config.Lx * y
                node_features.append(
                    [
                        site_data["n_up"][i],
                        site_data["n_dn"][i],
                        site_data["D_site"][i],
                        site_data["Sz_site"][i],
                        1.0 if (x + y) % 2 == 0 else -1.0,
                    ]
                )
        sample = build_graph_sample(
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            site_features=node_features,
            bond_strengths=bond_strengths,
            global_feats=[self.config.U, self.config.mu, float(self.config.Lx * self.config.Ly)],
            label=classify_phase_rule(self.config.U, self.get_observables().n, self.get_observables().Ms2),
            metadata={"runtime": True},
            max_nodes=self.config.Lx * self.config.Ly,
        )
        return sample.to_dict()

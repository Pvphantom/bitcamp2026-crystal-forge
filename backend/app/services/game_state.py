from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from app.domain.problem_spec import ProblemSpec
from app.domain.models import (
    AdaptiveMeasurementPlanResponse,
    AdaptiveMeasurementStepResponse,
    BondSnapshot,
    CreateStateRequest,
    EvolveRequest,
    ExportStateResponse,
    LatticeSnapshot,
    MLQProbePredictionResponse,
    MeasurementGroupResponse,
    MeasurementLibraryResponse,
    MeasurementPlanResponse,
    MetricsSummaryResponse,
    ObservablesResponse,
    PhasePredictionResponse,
    PlaceConfigurationRequest,
    QProbeRequest,
    SetParamsRequest,
    SiteSnapshot,
    TrustComparisonResponse,
    TrustMetricsResponse,
    TrustPredictionResponse,
)
from app.ml.infer import (
    MetricsReader,
    PhaseInferenceEngine,
    QProbeInferenceEngine,
    TrustInferenceEngine,
    TrustMetricsReader,
)
from app.ml.schema import build_graph_sample, classify_phase_rule
from app.observables.registry import build_default_observable_registry
from app.optimization.measurement_plan import (
    AdaptiveMeasurementStep,
    search_adaptive_measurement_plan,
    search_minimal_measurement_plan,
)
from app.solvers.exact_ed import ExactEDSolver
from app.solvers.mean_field import MeanFieldSolver
from app.solvers.registry import SolverRegistry
from app.physics.ed import expectation_value, ground_state, operator_matrix
from app.physics.hamiltonian import build_hamiltonian
from app.physics.lattice import nn_bonds
from app.physics.measurement_eval import NoiseModel
from app.physics.measurements import MeasurementGroup, build_measurement_library
from app.physics.measurements import explain_stop_reason
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
from app.analysis.solver_compare import compare_solver_results


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
        self.problem_spec: ProblemSpec | None = None
        self.observable_registry = build_default_observable_registry()
        self.solver_registry = SolverRegistry()
        self.solver_registry.register(ExactEDSolver(self.observable_registry))
        self.solver_registry.register(MeanFieldSolver())
        self.phase_inference = PhaseInferenceEngine()
        self.qprobe_inference = QProbeInferenceEngine()
        self.trust_inference = TrustInferenceEngine()
        self.metrics_reader = MetricsReader()
        self.trust_metrics_reader = TrustMetricsReader()
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

    def get_trust_metrics(self) -> TrustMetricsResponse:
        return TrustMetricsResponse(**self.trust_metrics_reader.summary())

    def evaluate_trust(self) -> TrustComparisonResponse:
        assert self.problem_spec is not None
        exact_result = self.solver_registry.get("exact_ed").solve(self.problem_spec)
        cheap_result = self.solver_registry.get("mean_field").solve(self.problem_spec)
        comparison = compare_solver_results(self.problem_spec, exact_result, cheap_result)
        trust_features = self._build_trust_features(cheap_result)
        trust_prediction = self.trust_inference.predict(trust_features)
        if trust_prediction is None:
            trust_response = TrustPredictionResponse(
                available=False,
                model_path=str(self.trust_inference.model_path),
                label=comparison.risk_label,
                confidence=None,
                predicted_max_abs_error=comparison.max_abs_error,
                recommended_action=self._trust_action(comparison.risk_label),
            )
            recommended_action = self._trust_action(comparison.risk_label)
        else:
            trust_response = TrustPredictionResponse(**trust_prediction)
            recommended_action = trust_prediction["recommended_action"]
        return TrustComparisonResponse(
            exact=self._observables_from_global_map(exact_result.global_observables),
            cheap_solver=self._observables_from_global_map(cheap_result.global_observables),
            abs_error=comparison.abs_error,
            rel_error=comparison.rel_error,
            max_abs_error=comparison.max_abs_error,
            energy_error=comparison.energy_error,
            risk_label=comparison.risk_label,
            trust_prediction=trust_response,
            recommended_action=recommended_action,
        )

    def get_qprobe_library(self) -> MeasurementLibraryResponse:
        library = build_measurement_library(self.config.Lx, self.config.Ly, self.config.t)
        return MeasurementLibraryResponse(
            observables={
                name: [self._measurement_group_response(group, [name]) for group in groups]
                for name, groups in library.items()
            }
        )

    def recommend_qprobe_plan(self, payload: QProbeRequest) -> MeasurementPlanResponse:
        self._ensure_state()
        assert self.statevector is not None
        observables = self.get_observables()
        result = search_minimal_measurement_plan(
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            t=self.config.t,
            state=self.statevector,
            target_observables=tuple(payload.targets),
            tolerance=payload.tolerance,
            shots_per_group=payload.shots_per_group,
            noise_model=NoiseModel(readout_flip_prob=payload.readout_flip_prob),
            seed=payload.seed,
        )
        qprobe_features = self._build_qprobe_features(payload, observables)
        ml_prediction = self.qprobe_inference.predict(qprobe_features)
        if ml_prediction is None:
            ml_response = MLQProbePredictionResponse(
                available=False,
                model_path=str(self.qprobe_inference.model_path),
            )
        else:
            ml_response = MLQProbePredictionResponse(
                **ml_prediction,
                matches_exact_cost=(ml_prediction["predicted_cost"] == result.recommended_plan.cost),
                matches_exact_success=(ml_prediction["predicted_success"] == result.success),
            )
        return MeasurementPlanResponse(
            success=result.success,
            targets=list(result.target_observables),
            tolerance=result.tolerance,
            full_cost=result.full_plan.cost,
            recommended_cost=result.recommended_plan.cost,
            measurement_savings=result.full_plan.cost - result.recommended_plan.cost,
            exact=result.exact,
            estimated=result.estimated,
            abs_error=result.abs_error,
            max_abs_error=result.max_abs_error,
            full_groups=[self._measurement_group_response(group) for group in result.full_plan.groups],
            recommended_groups=[
                self._measurement_group_response(group) for group in result.recommended_plan.groups
            ],
            ml_qprobe=ml_response,
            message=result.message,
        )

    def run_adaptive_qprobe(self, payload: QProbeRequest) -> AdaptiveMeasurementPlanResponse:
        self._ensure_state()
        assert self.statevector is not None
        result = search_adaptive_measurement_plan(
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            t=self.config.t,
            state=self.statevector,
            target_observables=tuple(payload.targets),
            tolerance=payload.tolerance,
            shots_per_group=payload.shots_per_group,
            noise_model=NoiseModel(readout_flip_prob=payload.readout_flip_prob),
            seed=payload.seed,
        )
        return AdaptiveMeasurementPlanResponse(
            success=result.success,
            targets=list(result.target_observables),
            tolerance=result.tolerance,
            full_cost=result.full_plan.cost,
            final_cost=result.final_plan.cost,
            measurement_savings=result.full_plan.cost - result.final_plan.cost,
            exact=result.exact,
            estimated=result.estimated,
            abs_error=result.abs_error,
            max_abs_error=result.max_abs_error,
            max_uncertainty=result.max_uncertainty,
            steps=[self._adaptive_step_response(step) for step in result.steps],
            message=f"{result.message} {explain_stop_reason(result.success, result.max_abs_error, result.max_uncertainty, result.tolerance)}",
        )

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
        self.problem_spec = ProblemSpec.hubbard(
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            t=self.config.t,
            U=self.config.U,
            mu=self.config.mu,
        )
        self.h_op = build_hamiltonian(
            self.config.Lx,
            self.config.Ly,
            self.config.t,
            self.config.U,
            self.config.mu,
        )
        self.h_matrix = operator_matrix(self.h_op)
        assert self.problem_spec is not None
        self.observable_ops = self.observable_registry.operator_map(self.problem_spec)
        self.observable_ops["bond_ops"] = self.observable_registry.hubbard_bond_operators(self.problem_spec)

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

    def _build_qprobe_features(
        self,
        payload: QProbeRequest,
        observables: ObservablesResponse,
    ) -> np.ndarray:
        target_flags = [1.0 if name in payload.targets else 0.0 for name in ["D", "n", "Ms2", "K", "Cs_max"]]
        return np.array(
            [
                self.config.U,
                self.config.mu,
                observables.D,
                observables.n,
                observables.Ms2,
                observables.K,
                observables.Cs_max,
                payload.tolerance,
                float(payload.shots_per_group),
                payload.readout_flip_prob,
                *target_flags,
            ],
            dtype=np.float32,
        )

    def _build_trust_features(self, cheap_result) -> np.ndarray:
        assert self.problem_spec is not None
        n_up = cheap_result.site_observables["n_up"]
        n_dn = cheap_result.site_observables["n_dn"]
        d_site = cheap_result.site_observables["D_site"]
        sz_site = cheap_result.site_observables["Sz_site"]
        abs_sz = [abs(value) for value in sz_site]
        density = [up + dn for up, dn in zip(n_up, n_dn, strict=True)]
        staggered_linear = 0.0
        for idx in range(len(sz_site)):
            x = idx % self.problem_spec.Lx
            y = idx // self.problem_spec.Lx
            staggered_linear += ((-1) ** (x + y)) * (n_up[idx] - n_dn[idx]) / self.problem_spec.nsites
        return np.array(
            [
                self.problem_spec.t,
                self.problem_spec.U,
                self.problem_spec.mu,
                cheap_result.global_observables["D"],
                cheap_result.global_observables["n"],
                cheap_result.global_observables["Ms2"],
                cheap_result.global_observables["K"],
                cheap_result.global_observables["Cs_max"],
                cheap_result.global_observables["energy"],
                float(sum(abs_sz) / len(abs_sz)),
                float(max(abs_sz)),
                float(staggered_linear),
                float(np.std(density)),
                float(np.std(sz_site)),
                float(np.std(d_site)),
                1.0 if cheap_result.metadata.get("converged", False) else 0.0,
                float(cheap_result.metadata.get("iterations", 0)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _observables_from_global_map(values: dict[str, float]) -> ObservablesResponse:
        return ObservablesResponse(
            D=float(values["D"]),
            n=float(values["n"]),
            Ms2=float(values["Ms2"]),
            K=float(values["K"]),
            Cs_max=float(values["Cs_max"]),
            energy=float(values["energy"]),
        )

    @staticmethod
    def _trust_action(label: str) -> str:
        if label == "safe":
            return "cheap_solver_ok"
        if label == "warning":
            return "check_exact_or_stronger_solver"
        return "escalate_to_exact_or_advanced_method"

    def _measurement_group_response(
        self,
        group: MeasurementGroup,
        supports_targets: list[str] | None = None,
    ) -> MeasurementGroupResponse:
        targets = supports_targets or self._supports_targets(group)
        return MeasurementGroupResponse(
            name=group.name,
            basis=group.basis,
            basis_label=group.basis_label,
            explanation=group.plain_english,
            supports_targets=targets,
            num_terms=group.num_terms,
            cost=group.cost,
        )

    def _adaptive_step_response(self, step: AdaptiveMeasurementStep) -> AdaptiveMeasurementStepResponse:
        return AdaptiveMeasurementStepResponse(
            step_index=step.step_index,
            chosen_group=self._measurement_group_response(step.chosen_group),
            current_cost=step.plan.cost,
            estimated=step.estimated,
            exact=step.exact,
            abs_error=step.abs_error,
            max_abs_error=step.max_abs_error,
            uncertainty=step.uncertainty,
            max_uncertainty=step.max_uncertainty,
        )

    def _supports_targets(self, group: MeasurementGroup) -> list[str]:
        supported = []
        library = build_measurement_library(self.config.Lx, self.config.Ly, self.config.t)
        term_paulis = {term.pauli for term in group.terms}
        for target, groups in library.items():
            target_terms = {term.pauli for target_group in groups for term in target_group.terms}
            if term_paulis & target_terms:
                supported.append(target)
        return supported

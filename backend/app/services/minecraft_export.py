from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.models import (
    GenericAnalysisResponse,
    GenericProblemRequest,
    GenericRoutingResponse,
    GenericSolverResultResponse,
    MinecraftBondRenderResponse,
    MinecraftBondValueResponse,
    MinecraftExportResponse,
    MinecraftMeasurementResponse,
    MinecraftObservablesResponse,
    MinecraftProblemInputsResponse,
    MinecraftProblemResponse,
    MinecraftRegimeResponse,
    MinecraftRoutingResponse,
    MinecraftSceneResponse,
    MinecraftSiteRenderResponse,
    MinecraftSiteValueResponse,
    MinecraftSolversResponse,
    MinecraftSolverSummaryResponse,
    MinecraftTrustResponse,
    MinecraftVisualizationHintsResponse,
    MinecraftWorkflowResponse,
)
from app.services.workflow import WorkflowService


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sign(value: float) -> int:
    if value > 1e-9:
        return 1
    if value < -1e-9:
        return -1
    return 0


@dataclass
class MinecraftExportService:
    workflow_service: WorkflowService
    _update_id: int = 0

    def export(self, payload: GenericProblemRequest) -> MinecraftExportResponse:
        analysis = self.workflow_service.analyze(payload)
        self._update_id += 1
        routing = self._routing_response(analysis.routing)
        active_solver = self._active_solver_result(analysis)
        site_primary_key, site_secondary_key, bond_primary_key = self._visual_keys(payload.model_family)
        return MinecraftExportResponse(
            schema_version="minecraft_v1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            update_id=self._update_id,
            scene=MinecraftSceneResponse(
                world_mode="superflat",
                layout="control_room_v1",
                origin={"x": 0, "y": 64, "z": 0},
            ),
            problem=MinecraftProblemResponse(
                model_family=payload.model_family,
                lattice={"Lx": payload.Lx, "Ly": payload.Ly, "nsites": payload.Lx * payload.Ly},
                inputs=MinecraftProblemInputsResponse(
                    parameters=dict(analysis.parameters),
                    qprobe_targets=self._requested_target_names(payload),
                    qprobe_tolerance=payload.qprobe_tolerance,
                    qprobe_shots_per_group=payload.qprobe_shots_per_group,
                    qprobe_readout_flip_prob=payload.qprobe_readout_flip_prob,
                    qprobe_seed=payload.qprobe_seed,
                ),
            ),
            routing=routing,
            workflow=MinecraftWorkflowResponse(
                decision_source="routing_model" if analysis.routing is not None else "legacy_fallback",
                active_path_type=self._active_path_type(analysis.workflow_decision.measurement_mode),
                selected_cheap_solver=analysis.selected_cheap_solver,
                selected_strong_solver=analysis.selected_strong_solver,
                active_solver=analysis.workflow_decision.active_solver,
                escalation_triggered=analysis.workflow_decision.escalation_triggered,
                measurement_mode=analysis.workflow_decision.measurement_mode,
                route_label=analysis.workflow_decision.route_label,
                recommendation=analysis.workflow_decision.recommendation,
            ),
            regime=MinecraftRegimeResponse(available=False),
            observables=MinecraftObservablesResponse(
                **{
                    "global": {key: float(value) for key, value in active_solver.observables.items()},
                    "site_values": self._site_values(payload, active_solver),
                    "bond_values": self._bond_values(payload.model_family, active_solver),
                }
            ),
            trust=MinecraftTrustResponse(
                risk_label=analysis.trust.risk_label,
                recommended_action=analysis.trust.recommended_action,
                max_abs_error=float(analysis.trust.max_abs_error),
                energy_error=float(analysis.trust.energy_error),
                abs_error={key: float(value) for key, value in analysis.trust.abs_error.items()},
                rel_error={key: float(value) for key, value in analysis.trust.rel_error.items()},
            ),
            solvers=MinecraftSolversResponse(
                cheap_solver=self._solver_summary(analysis.cheap_solver),
                exact_solver=self._solver_summary(analysis.exact_solver),
                strong_solver=self._solver_summary(analysis.strong_solver),
            ),
            measurement=MinecraftMeasurementResponse(
                enabled=analysis.qprobe_exact is not None or analysis.qprobe_adaptive is not None,
                planning_state_solver=(
                    analysis.qprobe_exact.planning_state_solver
                    if analysis.qprobe_exact is not None
                    else analysis.qprobe_adaptive.planning_state_solver
                    if analysis.qprobe_adaptive is not None
                    else None
                ),
                oracle_reference_solver=(
                    analysis.qprobe_exact.oracle_reference_solver
                    if analysis.qprobe_exact is not None
                    else analysis.qprobe_adaptive.oracle_reference_solver
                    if analysis.qprobe_adaptive is not None
                    else None
                ),
                qprobe=analysis.qprobe_exact,
                adaptive_qprobe=analysis.qprobe_adaptive,
            ),
            visualization_hints=MinecraftVisualizationHintsResponse(
                site_primary_key=site_primary_key,
                site_secondary_key=site_secondary_key,
                bond_primary_key=bond_primary_key,
                show_regime_panel=False,
                show_quantum_chamber=analysis.workflow_decision.measurement_mode == "quantum_follow_on",
                show_qprobe_ring=analysis.qprobe_exact is not None or analysis.qprobe_adaptive is not None,
                animate_site_updates=True,
                animate_bond_updates=True,
                animate_adaptive_steps=analysis.qprobe_adaptive is not None,
            ),
        )

    @staticmethod
    def _requested_target_names(payload: GenericProblemRequest) -> list[str]:
        target_names = list(payload.qprobe_targets)
        for spec in payload.qprobe_observable_specs:
            if spec.alias:
                target_names.append(spec.alias)
            elif spec.name:
                target_names.append(spec.name)
            else:
                target_names.append("custom_observable")
        return target_names

    @staticmethod
    def _routing_response(routing: GenericRoutingResponse | None) -> MinecraftRoutingResponse:
        if routing is None:
            return MinecraftRoutingResponse(
                available=False,
                abstained=True,
                abstain_reason="routing_unavailable",
            )
        confidence = None
        if routing.candidate_scores:
            confidence = float(max(routing.candidate_scores.values()))
        return MinecraftRoutingResponse(
            available=True,
            route_label=routing.route_label,
            confidence=confidence,
            recommended_action=routing.recommended_action,
            abstained=routing.abstained,
            abstain_reason=routing.abstain_reason,
            candidate_scores={key: float(value) for key, value in routing.candidate_scores.items()},
            intrinsic_label=routing.intrinsic_label,
            intrinsic_score=routing.intrinsic_score,
            intrinsic_reasons=list(routing.intrinsic_reasons),
        )

    @staticmethod
    def _active_path_type(measurement_mode: str) -> str:
        if measurement_mode == "quantum_follow_on":
            return "quantum"
        if measurement_mode == "oracle_fallback":
            return "exact_fallback"
        return "cheap"

    @staticmethod
    def _active_solver_result(analysis: GenericAnalysisResponse) -> GenericSolverResultResponse:
        active_solver_name = analysis.workflow_decision.active_solver
        if analysis.cheap_solver.solver_name == active_solver_name:
            return analysis.cheap_solver
        if analysis.strong_solver is not None and analysis.strong_solver.solver_name == active_solver_name:
            return analysis.strong_solver
        return analysis.exact_solver

    @staticmethod
    def _visual_keys(model_family: str) -> tuple[str, str | None, str]:
        if model_family == "tfim":
            return ("Mz", "Mx", "ZZ")
        return ("Sz", "double_occ", "spin_correlation")

    @staticmethod
    def _solver_summary(result: GenericSolverResultResponse | None) -> MinecraftSolverSummaryResponse:
        if result is None:
            return MinecraftSolverSummaryResponse(available=False)
        return MinecraftSolverSummaryResponse(
            available=True,
            solver_name=result.solver_name,
            energy=float(result.energy),
            observables={key: float(value) for key, value in result.observables.items()},
            site_observables={
                key: [float(value) for value in values]
                for key, values in result.site_observables.items()
            },
            bond_observables={key: float(value) for key, value in result.bond_observables.items()},
            metadata=result.metadata,
        )

    def _site_values(
        self,
        payload: GenericProblemRequest,
        result: GenericSolverResultResponse,
    ) -> list[MinecraftSiteValueResponse]:
        site_values: list[MinecraftSiteValueResponse] = []
        nsites = payload.Lx * payload.Ly
        for site_id in range(nsites):
            lattice_x = site_id % payload.Lx
            lattice_y = site_id // payload.Lx
            if payload.model_family == "tfim":
                mz = self._site_value(result, "Mz_site", site_id)
                mx = self._site_value(result, "Mx_site", site_id)
                site_values.append(
                    MinecraftSiteValueResponse(
                        site_id=site_id,
                        lattice_x=lattice_x,
                        lattice_y=lattice_y,
                        values={"Mz": mz, "Mx": mx},
                        render=MinecraftSiteRenderResponse(
                            primary_key="Mz",
                            primary_magnitude=_clip_unit(abs(mz)),
                            primary_sign=_sign(mz),
                            secondary_key="Mx",
                            secondary_magnitude=_clip_unit(abs(mx)),
                        ),
                    )
                )
                continue

            n_up = self._site_value(result, "n_up", site_id)
            n_dn = self._site_value(result, "n_dn", site_id)
            double_occ = self._site_value(result, "D_site", site_id)
            sz = self._site_value(result, "Sz_site", site_id)
            site_values.append(
                MinecraftSiteValueResponse(
                    site_id=site_id,
                    lattice_x=lattice_x,
                    lattice_y=lattice_y,
                    values={
                        "n_up": n_up,
                        "n_dn": n_dn,
                        "double_occ": double_occ,
                        "sz": sz,
                    },
                    render=MinecraftSiteRenderResponse(
                        primary_key="Sz",
                        primary_magnitude=_clip_unit(abs(sz) / 0.5),
                        primary_sign=_sign(sz),
                        secondary_key="double_occ",
                        secondary_magnitude=_clip_unit(double_occ),
                        occupancy_label=self._occupancy_label(n_up, n_dn),
                    ),
                )
            )
        return site_values

    @staticmethod
    def _site_value(result: GenericSolverResultResponse, key: str, index: int) -> float:
        values = result.site_observables.get(key, [])
        if index >= len(values):
            return 0.0
        return float(values[index])

    @staticmethod
    def _occupancy_label(n_up: float, n_dn: float) -> str:
        weights = {
            "empty": max(0.0, (1.0 - n_up) * (1.0 - n_dn)),
            "up": max(0.0, n_up * (1.0 - n_dn)),
            "down": max(0.0, (1.0 - n_up) * n_dn),
            "double": max(0.0, n_up * n_dn),
        }
        return max(weights.items(), key=lambda item: item[1])[0]

    def _bond_values(
        self,
        model_family: str,
        result: GenericSolverResultResponse,
    ) -> list[MinecraftBondValueResponse]:
        kind = "ZZ" if model_family == "tfim" else "spin_correlation"
        scale = 1.0 if model_family == "tfim" else 0.25
        bonds: list[MinecraftBondValueResponse] = []
        for key, value in sorted(result.bond_observables.items()):
            i_str, j_str = key.split("-", maxsplit=1)
            numeric_value = float(value)
            bonds.append(
                MinecraftBondValueResponse(
                    i=int(i_str),
                    j=int(j_str),
                    kind=kind,
                    value=numeric_value,
                    render=MinecraftBondRenderResponse(
                        magnitude=_clip_unit(abs(numeric_value) / scale),
                        sign=_sign(numeric_value),
                    ),
                )
            )
        return bonds

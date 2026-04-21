from pydantic import BaseModel, Field, model_validator


class CreateStateRequest(BaseModel):
    Lx: int = Field(default=2, ge=2)
    Ly: int = Field(default=2, ge=2)
    t: float = 1.0
    U: float = 4.0
    mu: float = 2.0


class SetParamsRequest(BaseModel):
    U: float | None = None
    mu: float | None = None
    t: float | None = None


class OccupationEntry(BaseModel):
    x: int
    y: int
    spin: str
    occupied: bool


class PlaceConfigurationRequest(BaseModel):
    default_state: str = Field(default="neel")
    occupations: list[OccupationEntry] = Field(default_factory=list)


class EvolveRequest(BaseModel):
    dt: float = 0.2
    steps: int = Field(default=1, ge=1)


class SiteSnapshot(BaseModel):
    i: int
    x: int
    y: int
    n_up: float
    n_dn: float
    double_occ: float
    sz: float


class BondSnapshot(BaseModel):
    i: int
    j: int
    strength: float


class LatticeSnapshot(BaseModel):
    Lx: int
    Ly: int
    sites: list[SiteSnapshot]
    bonds: list[BondSnapshot]


class ObservablesResponse(BaseModel):
    D: float
    n: float
    Ms2: float
    K: float
    Cs_max: float
    energy: float


class ModelStatusResponse(BaseModel):
    source: str
    model_loaded: bool
    model_path: str


class PhasePredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_status: ModelStatusResponse


class MetricsSummaryResponse(BaseModel):
    available: bool
    model_loaded: bool
    metrics_path: str
    model_path: str
    phase_labels: list[str] = Field(default_factory=list)
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    test_accuracy: float | None = None
    cross_lattice_accuracy: float | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    cross_lattice_confusion_matrix: list[list[int]] = Field(default_factory=list)


class TrustPredictionResponse(BaseModel):
    available: bool
    model_path: str
    label: str | None = None
    confidence: float | None = None
    predicted_max_abs_error: float | None = None
    recommended_action: str | None = None


class RoutingPredictionResponse(BaseModel):
    available: bool
    model_path: str
    label: str | None = None
    confidence: float | None = None
    recommended_action: str | None = None
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    abstain_reason: str | None = None


class TrustMetricsResponse(BaseModel):
    available: bool
    model_loaded: bool
    metrics_path: str
    model_path: str
    labels: list[str] = Field(default_factory=list)
    train_risk_accuracy: float | None = None
    val_risk_accuracy: float | None = None
    test_risk_accuracy: float | None = None
    test_error_mae: float | None = None
    test_false_safe_rate: float | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    cross_lattice_risk_accuracy: float | None = None
    cross_lattice_false_safe_rate: float | None = None


class TrustComparisonResponse(BaseModel):
    exact: ObservablesResponse
    cheap_solver: ObservablesResponse
    abs_error: dict[str, float]
    rel_error: dict[str, float]
    max_abs_error: float
    energy_error: float
    risk_label: str
    trust_prediction: TrustPredictionResponse
    recommended_action: str


class ObservablePauliTermRequest(BaseModel):
    pauli: str
    coeff_real: float = 1.0
    coeff_imag: float = 0.0


class ObservableTargetSpecRequest(BaseModel):
    name: str | None = None
    alias: str | None = None
    description: str | None = None
    pauli_terms: list[ObservablePauliTermRequest] = Field(default_factory=list)


class QProbeRequest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["D", "Ms2", "Cs_max"])
    observable_specs: list[ObservableTargetSpecRequest] = Field(default_factory=list)
    tolerance: float = Field(default=0.03, gt=0.0)
    shots_per_group: int = Field(default=4000, ge=1)
    readout_flip_prob: float = Field(default=0.0, ge=0.0, le=0.5)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_combined_target_budget(self) -> "QProbeRequest":
        if len(self.targets) + len(self.observable_specs) > 5:
            raise ValueError("QProbe requests are limited to at most 5 target operators per call")
        return self


class MeasurementGroupResponse(BaseModel):
    name: str
    basis: str
    basis_label: str
    explanation: str
    supports_targets: list[str] = Field(default_factory=list)
    num_terms: int
    cost: int


class MeasurementLibraryResponse(BaseModel):
    observables: dict[str, list[MeasurementGroupResponse]]


class MLQProbePredictionResponse(BaseModel):
    available: bool
    model_path: str
    predicted_cost: int | None = None
    predicted_success: bool | None = None
    predicted_error: float | None = None
    matches_exact_cost: bool | None = None
    matches_exact_success: bool | None = None


class QProbeModelPredictionResponse(BaseModel):
    available: bool
    model_path: str
    targets: list[str]
    full_cost: int
    predicted_cost: int | None = None
    full_gate_cost: int | None = None
    predicted_gate_cost: int | None = None
    predicted_success: bool | None = None
    predicted_error: float | None = None
    message: str


class MeasurementPlanResponse(BaseModel):
    success: bool
    targets: list[str]
    tolerance: float
    planning_state_solver: str | None = None
    oracle_reference_solver: str | None = None
    full_cost: int
    recommended_cost: int
    measurement_savings: int
    exact: dict[str, float]
    estimated: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    full_groups: list[MeasurementGroupResponse]
    recommended_groups: list[MeasurementGroupResponse]
    ml_qprobe: MLQProbePredictionResponse
    message: str


class AdaptiveMeasurementStepResponse(BaseModel):
    step_index: int
    chosen_group: MeasurementGroupResponse
    current_cost: int
    covered_targets: list[str]
    unresolved_targets: list[str]
    estimated: dict[str, float]
    exact: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    uncertainty: dict[str, float]
    max_uncertainty: float


class AdaptiveMeasurementPlanResponse(BaseModel):
    success: bool
    targets: list[str]
    tolerance: float
    runtime_stop_rule: str
    planning_state_solver: str | None = None
    oracle_reference_solver: str | None = None
    full_cost: int
    final_cost: int
    measurement_savings: int
    exact: dict[str, float]
    estimated: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    max_uncertainty: float
    oracle_benchmark_within_tolerance: bool
    steps: list[AdaptiveMeasurementStepResponse]
    message: str


class ExportStateResponse(BaseModel):
    lattice: LatticeSnapshot
    observables: ObservablesResponse
    phase: PhasePredictionResponse
    metrics: MetricsSummaryResponse


class GenericProblemRequest(BaseModel):
    model_family: str = Field(default="hubbard")
    Lx: int = Field(default=2, ge=2)
    Ly: int = Field(default=2, ge=2)
    parameters: dict[str, float] = Field(default_factory=dict)
    qprobe_targets: list[str] = Field(default_factory=list)
    qprobe_observable_specs: list[ObservableTargetSpecRequest] = Field(default_factory=list)
    qprobe_tolerance: float = Field(default=0.03, gt=0.0)
    qprobe_shots_per_group: int = Field(default=4000, ge=1)
    qprobe_readout_flip_prob: float = Field(default=0.0, ge=0.0, le=0.5)
    qprobe_seed: int | None = None

    @model_validator(mode="after")
    def validate_qprobe_target_budget(self) -> "GenericProblemRequest":
        if len(self.qprobe_targets) + len(self.qprobe_observable_specs) > 5:
            raise ValueError("Workflow QProbe requests are limited to at most 5 target operators per call")
        return self


class GenericSolverResultResponse(BaseModel):
    solver_name: str
    energy: float
    observables: dict[str, float]
    site_observables: dict[str, list[float]] = Field(default_factory=dict)
    bond_observables: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class GenericTrustResponse(BaseModel):
    abs_error: dict[str, float]
    rel_error: dict[str, float]
    max_abs_error: float
    energy_error: float
    risk_label: str
    recommended_action: str


class GenericRoutingResponse(BaseModel):
    route_label: str
    recommended_action: str
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
    abstain_reason: str | None = None
    intrinsic_label: str | None = None
    intrinsic_score: float | None = None
    intrinsic_reasons: list[str] = Field(default_factory=list)


class WorkflowDecisionResponse(BaseModel):
    escalation_triggered: bool
    active_solver: str
    measurement_mode: str
    recommendation: str
    route_label: str | None = None


class RoutingEvaluationResponse(BaseModel):
    model_family: str
    lattice: dict[str, int]
    parameters: dict[str, float]
    available_solvers: list[str] = Field(default_factory=list)
    selected_cheap_solver: str
    selected_strong_solver: str | None = None
    workflow_decision: WorkflowDecisionResponse
    routing: GenericRoutingResponse | None = None


class GenericAnalysisResponse(BaseModel):
    model_family: str
    lattice: dict[str, int]
    parameters: dict[str, float]
    available_solvers: list[str] = Field(default_factory=list)
    selected_cheap_solver: str
    selected_strong_solver: str | None = None
    workflow_decision: WorkflowDecisionResponse
    exact_solver: GenericSolverResultResponse
    cheap_solver: GenericSolverResultResponse
    strong_solver: GenericSolverResultResponse | None = None
    trust: GenericTrustResponse
    routing: GenericRoutingResponse | None = None
    measurement_library: MeasurementLibraryResponse
    qprobe_exact: MeasurementPlanResponse | None = None
    qprobe_adaptive: AdaptiveMeasurementPlanResponse | None = None


class MinecraftSceneResponse(BaseModel):
    world_mode: str
    layout: str
    origin: dict[str, int]


class MinecraftProblemInputsResponse(BaseModel):
    parameters: dict[str, float]
    qprobe_targets: list[str] = Field(default_factory=list)
    qprobe_tolerance: float
    qprobe_shots_per_group: int
    qprobe_readout_flip_prob: float
    qprobe_seed: int | None = None


class MinecraftProblemResponse(BaseModel):
    model_family: str
    lattice: dict[str, int]
    inputs: MinecraftProblemInputsResponse


class MinecraftRoutingResponse(BaseModel):
    available: bool
    route_label: str | None = None
    confidence: float | None = None
    recommended_action: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    intrinsic_label: str | None = None
    intrinsic_score: float | None = None
    intrinsic_reasons: list[str] = Field(default_factory=list)


class MinecraftWorkflowResponse(BaseModel):
    decision_source: str
    active_path_type: str
    selected_cheap_solver: str
    selected_strong_solver: str | None = None
    active_solver: str
    escalation_triggered: bool
    measurement_mode: str
    route_label: str | None = None
    recommendation: str


class MinecraftRegimeResponse(BaseModel):
    available: bool
    label: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class MinecraftSiteRenderResponse(BaseModel):
    primary_key: str
    primary_magnitude: float
    primary_sign: int
    secondary_key: str | None = None
    secondary_magnitude: float | None = None
    occupancy_label: str | None = None


class MinecraftSiteValueResponse(BaseModel):
    site_id: int
    lattice_x: int
    lattice_y: int
    values: dict[str, float]
    render: MinecraftSiteRenderResponse


class MinecraftBondRenderResponse(BaseModel):
    magnitude: float
    sign: int


class MinecraftBondValueResponse(BaseModel):
    i: int
    j: int
    kind: str
    value: float
    render: MinecraftBondRenderResponse


class MinecraftObservablesResponse(BaseModel):
    global_: dict[str, float] = Field(alias="global")
    site_values: list[MinecraftSiteValueResponse] = Field(default_factory=list)
    bond_values: list[MinecraftBondValueResponse] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class MinecraftTrustResponse(BaseModel):
    risk_label: str
    recommended_action: str
    max_abs_error: float
    energy_error: float
    abs_error: dict[str, float]
    rel_error: dict[str, float]


class MinecraftSolverSummaryResponse(BaseModel):
    available: bool = True
    solver_name: str | None = None
    energy: float | None = None
    observables: dict[str, float] = Field(default_factory=dict)
    site_observables: dict[str, list[float]] = Field(default_factory=dict)
    bond_observables: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class MinecraftSolversResponse(BaseModel):
    cheap_solver: MinecraftSolverSummaryResponse
    exact_solver: MinecraftSolverSummaryResponse
    strong_solver: MinecraftSolverSummaryResponse


class MinecraftMeasurementResponse(BaseModel):
    enabled: bool
    planning_state_solver: str | None = None
    oracle_reference_solver: str | None = None
    qprobe: MeasurementPlanResponse | None = None
    adaptive_qprobe: AdaptiveMeasurementPlanResponse | None = Field(default=None, alias="adaptive_qprobe")

    model_config = {"populate_by_name": True}


class MinecraftVisualizationHintsResponse(BaseModel):
    site_primary_key: str
    site_secondary_key: str | None = None
    bond_primary_key: str
    show_regime_panel: bool
    show_quantum_chamber: bool
    show_qprobe_ring: bool
    animate_site_updates: bool
    animate_bond_updates: bool
    animate_adaptive_steps: bool


class MinecraftExportResponse(BaseModel):
    schema_version: str
    timestamp: str
    update_id: int
    scene: MinecraftSceneResponse
    problem: MinecraftProblemResponse
    routing: MinecraftRoutingResponse
    workflow: MinecraftWorkflowResponse
    regime: MinecraftRegimeResponse
    observables: MinecraftObservablesResponse
    trust: MinecraftTrustResponse
    solvers: MinecraftSolversResponse
    measurement: MinecraftMeasurementResponse
    visualization_hints: MinecraftVisualizationHintsResponse

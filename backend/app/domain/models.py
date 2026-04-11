from pydantic import BaseModel, Field


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


class QProbeRequest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["D", "Ms2", "Cs_max"])
    tolerance: float = Field(default=0.03, gt=0.0)
    shots_per_group: int = Field(default=4000, ge=1)
    readout_flip_prob: float = Field(default=0.0, ge=0.0, le=0.5)
    seed: int | None = None


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


class MeasurementPlanResponse(BaseModel):
    success: bool
    targets: list[str]
    tolerance: float
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
    full_cost: int
    final_cost: int
    measurement_savings: int
    exact: dict[str, float]
    estimated: dict[str, float]
    abs_error: dict[str, float]
    max_abs_error: float
    max_uncertainty: float
    steps: list[AdaptiveMeasurementStepResponse]
    message: str


class ExportStateResponse(BaseModel):
    lattice: LatticeSnapshot
    observables: ObservablesResponse
    phase: PhasePredictionResponse
    metrics: MetricsSummaryResponse

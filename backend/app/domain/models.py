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


class ExportStateResponse(BaseModel):
    lattice: LatticeSnapshot
    observables: ObservablesResponse
    phase: PhasePredictionResponse
    metrics: MetricsSummaryResponse

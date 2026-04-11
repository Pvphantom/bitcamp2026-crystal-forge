from fastapi import APIRouter

from app.domain.models import (
    AdaptiveMeasurementPlanResponse,
    CreateStateRequest,
    EvolveRequest,
    ExportStateResponse,
    MeasurementLibraryResponse,
    MeasurementPlanResponse,
    MetricsSummaryResponse,
    ObservablesResponse,
    PhasePredictionResponse,
    PlaceConfigurationRequest,
    QProbeRequest,
    SetParamsRequest,
    TrustComparisonResponse,
    TrustMetricsResponse,
)
from app.services.game_state import HubbardGameStateService

router = APIRouter()
service = HubbardGameStateService()


@router.post("/state/create", response_model=ExportStateResponse)
def create_state(payload: CreateStateRequest) -> ExportStateResponse:
    return service.create_state(payload)


@router.post("/state/set-params", response_model=ExportStateResponse)
def set_params(payload: SetParamsRequest) -> ExportStateResponse:
    return service.set_params(payload)


@router.post("/state/reset-neel", response_model=ExportStateResponse)
def reset_neel() -> ExportStateResponse:
    return service.reset_to_neel()


@router.post("/state/place-configuration", response_model=ExportStateResponse)
def place_configuration(payload: PlaceConfigurationRequest) -> ExportStateResponse:
    return service.place_configuration(payload)


@router.post("/state/evolve", response_model=ExportStateResponse)
def evolve_state(payload: EvolveRequest) -> ExportStateResponse:
    return service.evolve(payload)


@router.post("/state/ground-state", response_model=ExportStateResponse)
def set_ground_state() -> ExportStateResponse:
    return service.set_ground_state()


@router.get("/state/observables", response_model=ObservablesResponse)
def get_observables() -> ObservablesResponse:
    return service.get_observables()


@router.get("/state/predict-phase", response_model=PhasePredictionResponse)
def predict_phase() -> PhasePredictionResponse:
    return service.predict_phase()


@router.get("/state/export", response_model=ExportStateResponse)
def export_state() -> ExportStateResponse:
    return service.export_state()


@router.get("/ml/metrics", response_model=MetricsSummaryResponse)
def get_ml_metrics() -> MetricsSummaryResponse:
    return service.get_metrics()


@router.get("/trust/metrics", response_model=TrustMetricsResponse)
def get_trust_metrics() -> TrustMetricsResponse:
    return service.get_trust_metrics()


@router.post("/trust/evaluate", response_model=TrustComparisonResponse)
def evaluate_trust() -> TrustComparisonResponse:
    return service.evaluate_trust()


@router.get("/qprobe/library", response_model=MeasurementLibraryResponse)
def get_qprobe_library() -> MeasurementLibraryResponse:
    return service.get_qprobe_library()


@router.post("/qprobe/recommend-plan", response_model=MeasurementPlanResponse)
def recommend_qprobe_plan(payload: QProbeRequest) -> MeasurementPlanResponse:
    return service.recommend_qprobe_plan(payload)


@router.post("/qprobe/adaptive-plan", response_model=AdaptiveMeasurementPlanResponse)
def run_adaptive_qprobe(payload: QProbeRequest) -> AdaptiveMeasurementPlanResponse:
    return service.run_adaptive_qprobe(payload)

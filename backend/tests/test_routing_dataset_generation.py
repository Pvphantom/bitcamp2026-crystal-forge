from app.analysis.routing_dataset import benchmark_sample_to_dict
from app.analysis.routing_label import RoutingLabelConfig
from app.domain.problem_spec import ProblemSpec
from scripts.data_gen_routing import _build_registry, build_routing_sample


def test_build_routing_sample_uses_cheap_features_and_records_strong_provenance() -> None:
    registry = _build_registry()
    sample = build_routing_sample(
        problem=ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0),
        registry=registry,
        reference_solver="exact_ed",
        reference_quality="strong",
        policy=RoutingLabelConfig(),
        sample_id=1,
    )

    payload = benchmark_sample_to_dict(sample)
    assert payload["reference_quality"] == "strong"
    assert payload["route_label"] == "scalable_classical"
    assert payload["problem_metadata"]["cheap_solver"] == "mean_field"
    assert "mean_field" in payload["solver_outcomes"]
    assert "exact_ed" in payload["solver_outcomes"]
    assert set(payload["feature_groups"].keys()) == {
        "family_flags",
        "lattice_metadata",
        "model_parameters",
        "cheap_observables",
        "stability",
    }
    assert "abs_error" in payload["solver_outcomes"]["mean_field"]
    assert payload["solver_outcomes"]["mean_field"]["max_abs_error"] is not None


def test_build_routing_sample_abstains_on_weak_reference_by_default() -> None:
    registry = _build_registry()
    sample = build_routing_sample(
        problem=ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=2.0),
        registry=registry,
        reference_solver="mean_field",
        reference_quality="weak",
        policy=RoutingLabelConfig(),
        sample_id=2,
    )

    payload = benchmark_sample_to_dict(sample)
    assert payload["route_label"] == "uncertain"
    assert payload["reference_quality"] == "weak"
    assert "policy=abstain" in payload["label_source"]

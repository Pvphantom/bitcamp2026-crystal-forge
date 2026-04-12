from __future__ import annotations

from app.analysis.qprobe_superconductor_channelize import build_superconductor_channel_plan
from app.analysis.qprobe_superconductor_decompose import (
    decompose_operator_map_by_transverse_signature,
)
from app.analysis.qprobe_superconductor_workflow import (
    bundle_orbit_map,
    build_superconductor_workflow_plan,
)
from app.domain.problem_spec import ProblemSpec
from app.observables.registry import build_default_observable_registry
from scripts.benchmark_qprobe_superconductor import benchmark_superconductor


def test_superconductor_observables_registered() -> None:
    registry = build_default_observable_registry()
    names = set(registry.names_for_family("hubbard"))
    assert "Pair_nn" in names
    assert "Pair_span" in names


def test_superconductor_benchmark_runs() -> None:
    report = benchmark_superconductor(
        node_budget=32,
        tolerance=0.03,
        shots_per_group=2000,
        readout_flip_prob=0.02,
        quick=True,
    )
    assert report["total_requests"] > 0
    assert report["overall"]["bounded"]["count"] == report["total_requests"]
    assert report["overall"]["decomposed"]["count"] == report["total_requests"]


def test_superconductor_channel_plan_groups_targets() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=4.0, mu=1.0)
    operator_map = {name: registry.operator(name, problem) for name in ("n", "Ms2", "Pair_nn", "Pair_span")}
    plan = build_superconductor_channel_plan(operator_map=operator_map, targets=tuple(operator_map))
    assert set(plan.channels) == {"charge", "pair", "spin"}


def test_superconductor_decomposition_collapses_pairing_signatures() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=6.0, mu=1.5)
    operator_map = {name: registry.operator(name, problem) for name in ("K", "Pair_nn", "Pair_span")}
    bundle = decompose_operator_map_by_transverse_signature(
        operator_map=operator_map,
        targets=("K", "Pair_nn", "Pair_span"),
    )
    assert len(bundle.target_names) == 9
    assert bundle.avg_signature_groups_per_target >= 2.0


def test_superconductor_workflow_decomposes_pair_and_transport() -> None:
    registry = build_default_observable_registry()
    problem = ProblemSpec.hubbard(Lx=2, Ly=2, t=1.0, U=6.0, mu=1.5)
    operator_map = {name: registry.operator(name, problem) for name in ("n", "Ms2", "K", "Pair_nn", "Pair_span")}
    workflow = build_superconductor_workflow_plan(
        operator_map=operator_map,
        targets=tuple(operator_map),
    )
    assert "charge" in workflow.direct_channels
    assert "spin" in workflow.direct_channels
    assert "transport" in workflow.decomposed_channels
    assert "pair" in workflow.decomposed_channels
    assert workflow.total_work_units >= 7
    pair_orbits = bundle_orbit_map(workflow.decomposed_channels["pair"])
    assert len(pair_orbits) == 3

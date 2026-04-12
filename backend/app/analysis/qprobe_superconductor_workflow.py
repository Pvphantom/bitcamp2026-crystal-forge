from __future__ import annotations

from dataclasses import dataclass

from qiskit.quantum_info import SparsePauliOp

from app.analysis.qprobe_superconductor_channelize import (
    SuperconductorChannelPlan,
    build_superconductor_channel_plan,
)
from app.analysis.qprobe_superconductor_decompose import (
    DecomposedOperatorBundle,
    decompose_operator_map_by_transverse_signature,
    signature_orbit_key,
)


@dataclass(frozen=True)
class SuperconductorWorkflowPlan:
    channel_plan: SuperconductorChannelPlan
    decomposed_channels: dict[str, DecomposedOperatorBundle]
    direct_channels: dict[str, tuple[str, ...]]

    @property
    def total_work_units(self) -> int:
        return sum(len(bundle.target_names) for bundle in self.decomposed_channels.values()) + sum(
            len(targets) for targets in self.direct_channels.values()
        )

    @property
    def decomposed_orbit_count(self) -> int:
        total = 0
        for bundle in self.decomposed_channels.values():
            total += len(_bundle_orbits(bundle))
        return total


def build_superconductor_workflow_plan(
    *,
    operator_map: dict[str, SparsePauliOp],
    targets: tuple[str, ...],
) -> SuperconductorWorkflowPlan:
    channel_plan = build_superconductor_channel_plan(
        operator_map=operator_map,
        targets=targets,
    )
    decomposed_channels: dict[str, DecomposedOperatorBundle] = {}
    direct_channels: dict[str, tuple[str, ...]] = {}
    for channel_name, channel_targets in channel_plan.channels.items():
        if channel_name in {"pair", "transport"}:
            decomposed_channels[channel_name] = decompose_operator_map_by_transverse_signature(
                operator_map=operator_map,
                targets=channel_targets,
            )
        else:
            direct_channels[channel_name] = channel_targets
    return SuperconductorWorkflowPlan(
        channel_plan=channel_plan,
        decomposed_channels=decomposed_channels,
        direct_channels=direct_channels,
    )


def bundle_orbit_map(bundle: DecomposedOperatorBundle) -> dict[str, tuple[str, ...]]:
    return _bundle_orbits(bundle)


def _bundle_orbits(bundle: DecomposedOperatorBundle) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for target_name in bundle.target_names:
        base_name, signature = target_name.split("@", 1)
        orbit = signature_orbit_key(base_name, signature)
        grouped.setdefault(orbit, []).append(target_name)
    return {key: tuple(values) for key, values in sorted(grouped.items())}

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_minecraft_workflow_tfim_quantum_payload_shape() -> None:
    response = client.post(
        "/api/minecraft/workflow",
        json={
            "model_family": "tfim",
            "Lx": 2,
            "Ly": 2,
            "parameters": {"J": 1.0, "h": 0.8, "g": 0.0},
            "qprobe_targets": ["Mz", "ZZ_nn", "Mstag2"],
            "qprobe_tolerance": 0.03,
            "qprobe_shots_per_group": 4000,
            "qprobe_readout_flip_prob": 0.02,
            "qprobe_seed": 7,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["schema_version"] == "minecraft_v1"
    assert body["scene"]["layout"] == "control_room_v1"
    assert body["problem"]["model_family"] == "tfim"
    assert body["problem"]["lattice"] == {"Lx": 2, "Ly": 2, "nsites": 4}
    assert body["workflow"]["decision_source"] in {"routing_model", "legacy_fallback"}
    assert body["workflow"]["active_path_type"] in {"cheap", "quantum", "exact_fallback"}
    assert body["routing"]["available"] is True
    assert len(body["observables"]["site_values"]) == 4
    assert body["observables"]["site_values"][0]["render"]["primary_key"] == "Mz"
    assert body["observables"]["bond_values"][0]["kind"] == "ZZ"
    assert body["solvers"]["cheap_solver"]["solver_name"] == "tfim_mean_field"
    assert body["solvers"]["exact_solver"]["solver_name"] == "exact_ed"
    assert body["solvers"]["strong_solver"]["available"] is True
    assert body["solvers"]["strong_solver"]["solver_name"] == "vqe"
    if body["workflow"]["measurement_mode"] == "quantum_follow_on":
        assert body["measurement"]["enabled"] is True
        assert body["measurement"]["planning_state_solver"] == "vqe"
        assert body["visualization_hints"]["show_quantum_chamber"] is True


def test_minecraft_workflow_hubbard_payload_shape() -> None:
    response = client.post(
        "/api/minecraft/workflow",
        json={
            "model_family": "hubbard",
            "Lx": 2,
            "Ly": 2,
            "parameters": {"t": 1.0, "U": 4.0, "mu": 2.0},
            "qprobe_targets": ["D", "Ms2", "Cs_max"],
            "qprobe_tolerance": 0.03,
            "qprobe_shots_per_group": 2000,
            "qprobe_readout_flip_prob": 0.01,
            "qprobe_seed": 7,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["problem"]["model_family"] == "hubbard"
    assert body["observables"]["site_values"][0]["render"]["occupancy_label"] in {
        "empty",
        "up",
        "down",
        "double",
    }
    assert body["observables"]["bond_values"][0]["kind"] == "spin_correlation"
    assert body["solvers"]["cheap_solver"]["solver_name"] == "mean_field"
    assert body["solvers"]["exact_solver"]["solver_name"] == "exact_ed"
    assert body["solvers"]["strong_solver"]["available"] is False
    assert body["visualization_hints"]["site_primary_key"] == "Sz"
    assert body["visualization_hints"]["bond_primary_key"] == "spin_correlation"

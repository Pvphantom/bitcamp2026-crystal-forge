from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_export_state_shape() -> None:
    response = client.get("/api/state/export")
    body = response.json()
    assert response.status_code == 200
    assert body["lattice"]["Lx"] == 2
    assert body["lattice"]["Ly"] == 2
    assert len(body["lattice"]["sites"]) == 4
    assert "observables" in body
    assert "phase" in body
    assert "metrics" in body
    assert "model_status" in body["phase"]


def test_create_set_params_and_reset_neel_flow() -> None:
    create = client.post("/api/state/create", json={"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0})
    assert create.status_code == 200
    assert abs(create.json()["observables"]["n"] - 1.0) < 1e-10

    update = client.post("/api/state/set-params", json={"U": 7.0, "mu": 3.5})
    assert update.status_code == 200
    assert update.json()["phase"]["label"] in {
        "Metal",
        "Mott Insulator",
        "Antiferromagnet",
        "Singlet-rich",
    }

    reset = client.post("/api/state/reset-neel")
    assert reset.status_code == 200
    sites = reset.json()["lattice"]["sites"]
    assert sites[0]["n_up"] == 1.0
    assert sites[0]["n_dn"] == 0.0


def test_place_configuration_reinitializes_state() -> None:
    response = client.post(
        "/api/state/place-configuration",
        json={
            "default_state": "empty",
            "occupations": [
                {"x": 0, "y": 0, "spin": "up", "occupied": True},
                {"x": 1, "y": 1, "spin": "down", "occupied": True},
            ],
        },
    )
    assert response.status_code == 200
    sites = response.json()["lattice"]["sites"]
    assert sites[0]["n_up"] == 1.0
    assert sites[0]["n_dn"] == 0.0
    assert sites[3]["n_up"] == 0.0
    assert sites[3]["n_dn"] == 1.0


def test_evolve_changes_energy_or_state_snapshot() -> None:
    client.post("/api/state/reset-neel")
    before = client.get("/api/state/export").json()
    after = client.post("/api/state/evolve", json={"dt": 0.1, "steps": 2}).json()
    assert after["observables"]["energy"] == after["observables"]["energy"]
    assert before["lattice"]["sites"] != after["lattice"]["sites"]


def test_ground_state_endpoint_lowers_energy_from_neel_state() -> None:
    client.post("/api/state/reset-neel")
    before = client.get("/api/state/observables").json()["energy"]
    after = client.post("/api/state/ground-state").json()["observables"]["energy"]
    assert after <= before + 1e-10


def test_ml_metrics_endpoint_returns_summary() -> None:
    response = client.get("/api/ml/metrics")
    body = response.json()
    assert response.status_code == 200
    assert "available" in body
    assert "model_loaded" in body
    assert "metrics_path" in body
    assert "phase_labels" in body


def test_trust_metrics_endpoint_returns_summary() -> None:
    response = client.get("/api/trust/metrics")
    body = response.json()
    assert response.status_code == 200
    assert "available" in body
    assert "model_loaded" in body
    assert "metrics_path" in body
    assert "labels" in body


def test_trust_evaluate_returns_comparison_and_recommendation() -> None:
    client.post("/api/state/create", json={"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 4.0})
    response = client.post("/api/trust/evaluate")
    body = response.json()
    assert response.status_code == 200
    assert "exact" in body
    assert "cheap_solver" in body
    assert "abs_error" in body
    assert "risk_label" in body
    assert "trust_prediction" in body
    assert "recommended_action" in body
    assert body["risk_label"] in {"safe", "warning", "unsafe"}


def test_qprobe_library_endpoint_returns_measurement_groups() -> None:
    response = client.get("/api/qprobe/library")
    body = response.json()
    assert response.status_code == 200
    assert "observables" in body
    assert "D" in body["observables"]
    assert len(body["observables"]["D"]) >= 1


def test_qprobe_recommend_plan_compresses_z_only_targets() -> None:
    client.post("/api/state/reset-neel")
    response = client.post(
        "/api/qprobe/recommend-plan",
        json={
            "targets": ["D", "n", "Ms2", "Cs_max"],
            "tolerance": 0.03,
            "shots_per_group": 20000,
            "readout_flip_prob": 0.0,
            "seed": 11,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["recommended_cost"] <= body["full_cost"]
    assert body["measurement_savings"] >= 0
    assert body["max_abs_error"] <= body["tolerance"]
    assert "ml_qprobe" in body
    assert body["ml_qprobe"]["available"] is True
    assert body["ml_qprobe"]["predicted_cost"] is not None


def test_qprobe_recommend_plan_reports_failure_honestly() -> None:
    client.post("/api/state/reset-neel")
    response = client.post(
        "/api/qprobe/recommend-plan",
        json={
            "targets": ["D", "n", "Ms2", "K", "Cs_max"],
            "tolerance": 1e-4,
            "shots_per_group": 500,
            "readout_flip_prob": 0.08,
            "seed": 21,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["success"] is False
    assert body["recommended_cost"] == body["full_cost"]
    assert body["max_abs_error"] > body["tolerance"]
    assert "ml_qprobe" in body
    assert body["ml_qprobe"]["available"] is True


def test_adaptive_qprobe_stops_early_on_easy_case() -> None:
    client.post("/api/state/create", json={"Lx": 2, "Ly": 2, "t": 1.0, "U": 8.0, "mu": 4.0})
    client.post("/api/state/ground-state")
    response = client.post(
        "/api/qprobe/adaptive-plan",
        json={
            "targets": ["D", "n", "Ms2", "Cs_max"],
            "tolerance": 0.03,
            "shots_per_group": 4000,
            "readout_flip_prob": 0.02,
            "seed": 11,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["final_cost"] < body["full_cost"]
    assert len(body["steps"]) == body["final_cost"]


def test_adaptive_qprobe_uses_full_plan_on_hard_case() -> None:
    client.post("/api/state/create", json={"Lx": 2, "Ly": 2, "t": 1.0, "U": 4.0, "mu": 2.0})
    client.post("/api/state/ground-state")
    response = client.post(
        "/api/qprobe/adaptive-plan",
        json={
            "targets": ["D", "n", "Ms2", "K", "Cs_max"],
            "tolerance": 0.01,
            "shots_per_group": 2000,
            "readout_flip_prob": 0.08,
            "seed": 11,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["success"] is False
    assert body["final_cost"] == body["full_cost"]
    assert len(body["steps"]) == body["full_cost"]

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

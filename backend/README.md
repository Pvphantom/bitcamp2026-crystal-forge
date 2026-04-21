# Crystal Forge — Backend

FastAPI service that owns all physics, numerics, and ML for the project. The React
dashboard and Minecraft mod are thin clients; all state, routing decisions, and
observables are computed here.

- Python **3.11+**
- FastAPI + Uvicorn (ASGI)
- NumPy / SciPy / scikit-learn for classical numerics
- PyTorch for CorrMap + QProbe neural models
- Qiskit + qiskit-nature for VQE-style quantum-frontier solvers

---

## Layout

```
backend/
├── app/
│   ├── main.py              FastAPI app factory
│   ├── api/routes.py        REST endpoints (incl. /api/minecraft/workflow)
│   ├── core/                config, logging, schemas shared across modules
│   ├── domain/              Pydantic models for requests / responses / presets
│   ├── physics/             Hamiltonian builders (Fermi–Hubbard, TFIM)
│   ├── solvers/             Exact diag, mean-field, DMRG-lite, VQE wrappers
│   ├── observables/         Charge / spin / transport / pairing channels
│   ├── optimization/        Parameter sweeps, adaptive step control
│   ├── analysis/            Intrinsic-diagnostic feature builders (the 27-dim
│   │                        behavioral vector used by CorrMap)
│   ├── ml/                  CorrMap classifier + QProbe planner (PyTorch)
│   └── services/            Orchestration — binds solver → analysis → ML → API
├── scripts/                 Data generation, training, benchmarks, evaluation
├── tests/                   pytest suite
├── artifacts/               Trained model checkpoints and generated datasets
└── pyproject.toml
```

---

## Running locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# dev server with hot reload
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Quick checks:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/minecraft/workflow \
     -H 'Content-Type: application/json' \
     -d @tests/fixtures/tfim_default.json
```

---

## Core API

| Method | Path                          | Purpose                                                        |
|--------|-------------------------------|----------------------------------------------------------------|
| GET    | `/health`                     | Liveness probe                                                 |
| POST   | `/api/minecraft/workflow`     | Run full workflow → returns scene payload for Minecraft/React  |
| POST   | `/api/solve`                  | Run a single solver without ML routing                         |
| POST   | `/api/corrmap/predict`        | CorrMap routing verdict only                                   |
| POST   | `/api/qprobe/plan`            | Next-observable suggestion for a given posterior               |

All request/response schemas live in `app/domain/` — see those Pydantic models
for the source of truth.

### `POST /api/minecraft/workflow` — response shape

```jsonc
{
  "routing": {
    "verdict": "classical_scalable" | "quantum_frontier",
    "confidence": 0.87,
    "features": { /* 49-dim vector, named */ }
  },
  "observables": { "charge": ..., "spin": ..., "pairing": ... },
  "scene": {
    "origin": { "x": 0, "y": 64, "z": 0 },
    "lattice": [ /* per-site { pos, block_id, intensity } */ ],
    "controls": [ /* button blocks — model toggle, param ±, run */ ]
  }
}
```

---

## ML models

### CorrMap — binary solver router
- `app/ml/binary_corrmap_model.py` — `BinaryCorrMapMLP` (LayerNorm + GELU,
  2×96 hidden, sigmoid output).
- Input: 49-dim vector = 22 base physics features + 27 intrinsic diagnostics
  (`app/analysis/intrinsic_feature_vector.py` → `INTRINSIC_AUGMENTED_FEATURE_DIM`).
- Labels: `classical_scalable` vs `quantum_frontier`.
- Trained on 2×2, 4×4, 6×6 regime samples; **8×8 is held out**. Intrinsic
  features are behavioral (stability under perturbation, size-consistency,
  ansatz disagreement, hysteresis), so the learned decision boundary transfers
  to lattice sizes never seen in training.

### QProbe — adaptive measurement planner
Plans which observable channel (charge / spin / transport / pairing) to commit
the next batch of shots to, given the current posterior. Variants live under
`app/ml/` and `scripts/train_qprobe_*`.

---

## Scripts

`scripts/` is flat on purpose — each file is one runnable experiment. Naming:

- `data_gen_*` — generate a training dataset (writes to `artifacts/`)
- `train_*` — train a model checkpoint (writes to `artifacts/`)
- `eval_*` — evaluate a trained model on a benchmark
- `benchmark_*` — end-to-end benchmark runs
- `generate_*_report.py` — produce human-readable reports from a run

Typical flow:

```bash
python scripts/data_gen_intrinsic_corrmap_general.py
python scripts/train_binary_corrmap.py
python scripts/eval_binary_corrmap_on_regime_benchmark.py
```

---

## Testing

```bash
pytest -q
```

Tests cover Hamiltonian construction, observable consistency, and API contract
stability.

---

## Configuration

The backend reads no config file by default — all presets live in
`app/domain/presets.py`. Environment variables of interest:

| Variable                  | Default                             | Purpose                          |
|---------------------------|-------------------------------------|----------------------------------|
| `CRYSTALFORGE_LOG_LEVEL`  | `INFO`                              | Uvicorn / app log level          |
| `CRYSTALFORGE_MODEL_DIR`  | `backend/artifacts`                 | Where trained checkpoints live   |

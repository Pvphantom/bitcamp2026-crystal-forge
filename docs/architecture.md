# Architecture Plan

## Goal

Build a Python-first Hubbard simulation backend that can serve both:

- the hackathon web frontend
- a later Minecraft visualization layer

## Principles

- Keep all physics and ML logic in Python.
- Make the frontend a thin client.
- Define stable JSON payloads early so browser and Minecraft clients can share them.
- Get `2x2` exact simulation correct before adding VQE, `2x3`, or hardware work.

## Backend Modules

- `app/physics/lattice.py`
  - site indexing
  - nearest-neighbor bonds
  - spin-major qubit ordering helpers
- `app/physics/hamiltonian.py`
  - Hubbard Hamiltonian builder
  - acceptance-test helpers
- `app/physics/state_prep.py`
  - Neel state
  - product-state initialization
- `app/physics/observables.py`
  - `D`, `n`, `Ms2`, `K`, `Cs_max`, `Pd`
- `app/physics/ed.py`
  - exact diagonalization, full-space first
- `app/physics/vqe.py`
  - HVA and later fallback ansatzes
- `app/ml/`
  - dataset schema
  - inference
  - training scripts
- `app/services/game_state.py`
  - orchestrates state, parameters, observables, and phase prediction
- `app/api/routes.py`
  - HTTP surface for frontend and future external clients

## API Shape

Starter endpoints:

- `POST /api/state/create`
- `POST /api/state/reset-neel`
- `POST /api/state/place-configuration`
- `POST /api/state/set-params`
- `POST /api/state/evolve`
- `POST /api/state/ground-state`
- `GET /api/state/observables`
- `GET /api/state/predict-phase`
- `GET /api/state/export`

## Shared Export Schema

The backend should eventually return a shape like:

```json
{
  "lattice": {
    "Lx": 2,
    "Ly": 2,
    "sites": [],
    "bonds": []
  },
  "observables": {},
  "phase": {}
}
```

That payload is intentionally generic enough for:

- React tile rendering
- Minecraft block/entity rendering
- deck screenshots and offline exports

## Implementation Milestones

1. Scaffolding and repo split
2. Hamiltonian + acceptance tests
3. ED + observables
4. FastAPI state service
5. Frontend rewrite against backend
6. GNN training/inference
7. Stretch goals after the core is stable

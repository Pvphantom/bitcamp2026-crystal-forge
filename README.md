# Crystal Forge

The architecture is now:

- `backend/`: Python FastAPI service for Hubbard physics, observables, ML inference, and future Minecraft integration.
- `frontend/`: React + Vite client for rapid iteration on controls and visualization.
- `docs/`: specs and architecture notes.

## Current Status

## Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── domain/
│   │   ├── ml/
│   │   ├── physics/
│   │   ├── services/
│   │   └── main.py
│   ├── scripts/
│   ├── tests/
│   └── pyproject.toml
├── docs/
│   ├── architecture.md
│   └── specs/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Near-Term Build Order

1. Implement `backend/app/physics/hamiltonian.py`.
2. Add the 4 acceptance tests from the Hubbard spec.
3. Add exact diagonalization and core observables.
4. Expose them through FastAPI.
5. Replace the current frontend with a Hubbard lattice/material editor.
6. Add classifier training and inference after the physics core is trusted.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload
```

## Minecraft Direction

The backend is the long-term product core. The browser UI and any future Minecraft visualization is secondary to getting some display.

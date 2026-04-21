# Crystal Forge — Bitcamp 2026 MCQuant Superconductor

Crystal Forge is a **solver-routing and measurement-planning framework for strongly-correlated
electron systems** (Fermi–Hubbard, TFIM) with a **Minecraft Fabric mod** as an interactive
front end. A classical ML "CorrMap" classifier decides, per parameter regime, whether a
cheap mean-field / tensor-network solver is trustworthy or whether the problem belongs in
the *quantum frontier* that requires a VQE-style quantum treatment; a companion "QProbe"
planner then picks which observables to measure to maximize information per shot.

The repo has three deployable pieces:

| Subproject  | Stack                                 | Purpose                                                   |
|-------------|---------------------------------------|-----------------------------------------------------------|
| `backend/`  | Python 3.11, FastAPI, PyTorch, Qiskit | Physics solvers, ML models, REST API, training scripts    |
| `frontend/` | React 18, Vite                        | Dashboard for running workflows and inspecting results    |
| `minecraft/`| Java 17, Fabric 1.20.4                | In-game 3D lattice visualization + interactive controls   |

See each subproject's README for details.

---

## Architecture at a glance

```
                 ┌─────────────────────────────────────────┐
                 │           Crystal Forge backend         │
                 │        (FastAPI @ 127.0.0.1:8000)       │
                 │                                         │
  preset/params  │   physics/  →  solvers/  →  analysis/   │   JSON payload
  ─────────────▶ │         ↘  ml/ (CorrMap, QProbe) ↙      │ ────────────────▶
                 │                  ↓                      │   (scene +
                 │            api/routes.py                │    observables)
                 └─────────────────────────────────────────┘
                         ▲                         │
                         │                         │
                  ┌──────┴──────┐          ┌───────▼────────┐
                  │   React UI  │          │ Minecraft mod  │
                  │ (frontend/) │          │ (minecraft/)   │
                  └─────────────┘          └────────────────┘
```

The backend is the single source of truth. Both UIs talk to the same REST endpoint
(`POST /api/minecraft/workflow`) and receive a JSON payload containing a scene
description (lattice geometry, colored blocks per site, routing verdict, observables).

---

## Quick start

Clone and run all three layers locally.

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check: `curl http://127.0.0.1:8000/health`.

### 2. Frontend (optional — for dashboard)

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

### 3. Minecraft mod

```bash
cd minecraft
./gradlew runClient
```

On first player join the mod auto-fetches the default TFIM preset from the backend
and renders the lattice. Right-click the control wall to switch models or adjust
parameters — the scene live-updates.

---

## The science, briefly

* **Problem.** Exact diagonalization of the 2D Fermi–Hubbard model scales as
  `4^N`, so beyond ~14 sites it's intractable. Mean-field / DMFT / tensor-network
  solvers are cheap but silently wrong in strongly-correlated regimes
  (Mott, frustrated magnets, d-wave pairing). You can't know in advance which
  regime you're in.
* **CorrMap.** A binary MLP classifier trained on a 49-dim feature vector
  (22 base physics features + 27 *intrinsic diagnostics* — mean-field stability,
  sensitivity, size-consistency, ansatz disagreement, hysteresis).
  Labels `classical_scalable` vs `quantum_frontier`. Trained on 2×2 + 4×4 + 6×6
  regime samples; **held out on 8×8** — intrinsic features are behavioral, not
  size-specific, so the classifier generalizes to lattice sizes it has never seen.
* **QProbe.** An adaptive measurement planner that picks which observables
  (charge / spin / transport / pairing channels) to commit shots to next, given
  the current posterior.
* **Minecraft bridge.** The live 3D visualization turns an otherwise opaque
  solver decision into something a judge (or a collaborator) can *walk around
  inside* and perturb with a click.

For the full technical design, see `docs/` and the design document shared
out-of-band.

---

## Repository layout

```
bitcamp2026-mcquant-superconductor/
├── backend/      FastAPI service, solvers, ML models, training scripts
├── frontend/     React + Vite dashboard
├── minecraft/    Fabric 1.20.4 mod (Java 17)
├── docs/         design notes and internal memos
└── README.md     (this file)
```

## License

Unlicensed hackathon project — contact the authors before reuse.

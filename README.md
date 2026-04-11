# String Breaker ⚛️

A real-time quantum simulation game where you break the energy string between quarks by tuning physics parameters — built on the (1+1)D Schwinger model.

## What is this?

In particle physics, quarks are confined — you can never pull one out of a proton because the energy "string" between them grows until it snaps, creating new particles from empty space. This game lets you see that happen in real time using a genuine quantum simulation.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## How to Play

1. **Level Select** — work through 6 challenge levels or try Free Play
2. **Hit Play** — watch the quantum simulation evolve in real time  
3. **Drag sliders** — adjust string strength, particle weight, and background field
4. **Break the string** — when the inner charges hit the threshold, you win
5. **Earn stars** — faster breaks = more stars

## Tech Stack

- **Frontend**: React + Vite
- **Simulation**: Pure JavaScript implementation of the lattice Schwinger model
  - Kogut–Susskind staggered fermions
  - Jordan–Wigner transformation to qubits
  - Gauss's law elimination of gauge field
  - 6-site lattice → 64-dimensional Hilbert space
  - Exact time evolution via matrix exponential
- **Backend** (optional, in `backend/`): Python + Qiskit + FastAPI for hardware-ready quantum circuits

## Project Structure

```
string-breaker/
├── index.html          # Entry point
├── src/
│   ├── main.jsx        # React mount
│   ├── App.jsx         # Game + simulation engine
│   └── index.css       # Global styles
├── backend/            # Python quantum backend
│   ├── hamiltonian.py  # Hamiltonian builder (SparsePauliOp)
│   ├── observables.py  # State prep + measurement
│   ├── evolution.py    # Trotterized time evolution
│   ├── vqe.py          # Variational quantum eigensolver
│   ├── game_state.py   # Core game state class
│   └── server.py       # FastAPI server
├── package.json
└── vite.config.js
```

## The Physics

The Schwinger model is quantum electrodynamics in 1+1 dimensions — the simplest gauge theory that exhibits **confinement** (particles can't be separated) and **string breaking** (the vacuum produces new particle-antiparticle pairs when the energy string gets too taut).

We simulate it by:
1. Discretizing space onto a 6-site lattice with staggered fermions
2. Mapping fermions to qubits via the Jordan–Wigner transformation
3. Eliminating the gauge field using Gauss's law
4. Evolving the 64-dimensional state vector under the exact Hamiltonian

The result is a real quantum simulation running in your browser — every number you see is computed from the actual wavefunction.

## Bitcamp 2026 — Quantum Track

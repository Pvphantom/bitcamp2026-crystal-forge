# Crystal Forge — Frontend

React + Vite dashboard for running Crystal Forge workflows and inspecting
routing / observable output. Primarily a developer-facing surface — the
judge-facing visualization lives in the Minecraft mod.

- React **18**
- Vite **6**
- Plain CSS (no framework)
- Talks to the backend at `http://127.0.0.1:8000` by default

---

## Running

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run build     # production build → dist/
npm run preview   # serve the production build locally
```

Make sure the backend (`backend/`) is running on port 8000 first, or
adjust the API base in `src/` accordingly.

---

## Layout

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx        entry point
    ├── App.jsx         top-level dashboard
    └── index.css       styling
```

The dashboard currently exposes:

1. **Preset picker** — TFIM quantum-escalation, Hubbard fallback, etc.
2. **Parameter sliders** — J / h / g for TFIM, t / U / μ for Hubbard.
3. **Run** button → hits `POST /api/minecraft/workflow`.
4. **Result panels** — routing verdict + confidence, observable summary,
   raw payload viewer.

---

## Pointing at a remote backend

The API base URL is read from `src/App.jsx`. To point the dashboard at a
backend other than `127.0.0.1:8000`, edit that constant (or plumb it through
`import.meta.env` and a `.env.local` file — standard Vite pattern).

---

## Production build notes

`npm run build` produces a static bundle in `dist/` that can be served by any
static host. There is no SSR and no server-side secrets — the frontend only
consumes public backend endpoints.

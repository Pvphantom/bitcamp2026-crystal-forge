import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const PHASE_COLORS = {
  Metal: "#4fb3ff",
  "Mott Insulator": "#ff9d42",
  Antiferromagnet: "#67e39a",
  "Singlet-rich": "#ff6f91",
  unclassified: "#a4acc4",
};

const OCCUPANCY_ORDER = ["empty", "up", "down", "double"];

function occupancyFromSite(site) {
  if (site.n_up > 0.5 && site.n_dn > 0.5) return "double";
  if (site.n_up > 0.5) return "up";
  if (site.n_dn > 0.5) return "down";
  return "empty";
}

function cycleOccupancy(current) {
  const index = OCCUPANCY_ORDER.indexOf(current);
  return OCCUPANCY_ORDER[(index + 1) % OCCUPANCY_ORDER.length];
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json();
}

export default function App() {
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [params, setParams] = useState({ t: 1, U: 4, mu: 2 });
  const [boardMode, setBoardMode] = useState("neel");
  const [board, setBoard] = useState({});

  const syncBoardFromState = (nextState) => {
    const nextBoard = {};
    for (const site of nextState.lattice.sites) {
      nextBoard[`${site.x}:${site.y}`] = occupancyFromSite(site);
    }
    setBoard(nextBoard);
  };

  const loadExport = async () => {
    setLoading(true);
    setError("");
    try {
      const nextState = await request("/api/state/export");
      setState(nextState);
      setParams((current) => ({
        ...current,
        t: current.t,
        U: current.U,
        mu: current.mu,
      }));
      syncBoardFromState(nextState);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExport();
  }, []);

  useEffect(() => {
    if (!state) return;
    setParams((current) => ({
      ...current,
      t: current.t,
      U: current.U,
      mu: current.mu,
    }));
  }, [state]);

  const runAction = async (action) => {
    setPending(true);
    setError("");
    try {
      const nextState = await action();
      setState(nextState);
      syncBoardFromState(nextState);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const applyParams = async () => {
    await runAction(() =>
      request("/api/state/set-params", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    );
  };

  const applyBoard = async () => {
    const occupations = Object.entries(board).flatMap(([key, value]) => {
      const [x, y] = key.split(":").map(Number);
      if (value === "empty") return [];
      if (value === "up") {
        return [{ x, y, spin: "up", occupied: true }];
      }
      if (value === "down") {
        return [{ x, y, spin: "down", occupied: true }];
      }
      return [
        { x, y, spin: "up", occupied: true },
        { x, y, spin: "down", occupied: true },
      ];
    });

    await runAction(() =>
      request("/api/state/place-configuration", {
        method: "POST",
        body: JSON.stringify({
          default_state: boardMode,
          occupations,
        }),
      }),
    );
  };

  const boardSites = useMemo(() => state?.lattice?.sites ?? [], [state]);
  const phaseColor = PHASE_COLORS[state?.phase?.label] ?? PHASE_COLORS.unclassified;
  const modelStatus = state?.phase?.model_status;
  const metrics = state?.metrics;

  if (loading) {
    return (
      <main className="app-shell">
        <section className="hero-card loading-card">
          <p className="eyebrow">Crystal Forge</p>
          <h1>Connecting to the Hubbard backend</h1>
          <p>Waiting for the FastAPI service at <code>{API_BASE}</code>.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Crystal Forge</p>
          <h1>Design correlated materials on a small Hubbard lattice</h1>
          <p className="hero-copy">
            The browser is now a thin client. All physics, exact diagonalization,
            observables, and phase prediction live in the Python backend so this
            same contract can later drive a Minecraft visualization.
          </p>
        </div>
        <div className="hero-actions">
          <button onClick={loadExport} disabled={pending}>Refresh</button>
          <button onClick={() => runAction(() => request("/api/state/reset-neel", { method: "POST" }))} disabled={pending}>
            Reset N&eacute;el
          </button>
          <button onClick={() => runAction(() => request("/api/state/ground-state", { method: "POST" }))} disabled={pending}>
            Exact Ground State
          </button>
        </div>
      </section>

      {error ? (
        <section className="error-banner">
          <strong>Backend error:</strong> {error}
        </section>
      ) : null}

      <section className="dashboard-grid">
        <article className="panel lattice-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Lattice Editor</p>
              <h2>2x2 material board</h2>
            </div>
            <span className="status-pill">{pending ? "Updating" : "Live"}</span>
          </div>

          <div className="board-controls">
            <label>
              Default reset
              <select value={boardMode} onChange={(event) => setBoardMode(event.target.value)} disabled={pending}>
                <option value="neel">N&eacute;el</option>
                <option value="empty">Empty</option>
                <option value="polarized">Polarized</option>
              </select>
            </label>
            <button onClick={applyBoard} disabled={pending}>Apply Board</button>
          </div>

          <div className="site-grid">
            {boardSites.map((site) => {
              const key = `${site.x}:${site.y}`;
              const occupancy = board[key] ?? occupancyFromSite(site);
              return (
                <button
                  key={site.i}
                  className={`site-tile site-${occupancy}`}
                  onClick={() =>
                    setBoard((current) => ({
                      ...current,
                      [key]: cycleOccupancy(current[key] ?? occupancy),
                    }))
                  }
                  disabled={pending}
                >
                  <span className="site-label">
                    ({site.x}, {site.y})
                  </span>
                  <span className="site-state">{occupancy}</span>
                  <span className="site-values">
                    n↑ {site.n_up.toFixed(2)} · n↓ {site.n_dn.toFixed(2)}
                  </span>
                  <span className="site-values">
                    D {site.double_occ.toFixed(2)} · Sz {site.sz.toFixed(2)}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="legend-row">
            {OCCUPANCY_ORDER.map((entry) => (
              <span key={entry} className={`legend-chip chip-${entry}`}>{entry}</span>
            ))}
          </div>
        </article>

        <article className="panel controls-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Controls</p>
              <h2>Hamiltonian parameters</h2>
            </div>
          </div>

          <div className="slider-stack">
            <label>
              <span>t</span>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={params.t}
                onChange={(event) => setParams((current) => ({ ...current, t: Number(event.target.value) }))}
                disabled={pending}
              />
              <strong>{params.t.toFixed(1)}</strong>
            </label>
            <label>
              <span>U</span>
              <input
                type="range"
                min="0"
                max="10"
                step="0.1"
                value={params.U}
                onChange={(event) => setParams((current) => ({ ...current, U: Number(event.target.value) }))}
                disabled={pending}
              />
              <strong>{params.U.toFixed(1)}</strong>
            </label>
            <label>
              <span>&mu;</span>
              <input
                type="range"
                min="-2"
                max="5"
                step="0.1"
                value={params.mu}
                onChange={(event) => setParams((current) => ({ ...current, mu: Number(event.target.value) }))}
                disabled={pending}
              />
              <strong>{params.mu.toFixed(1)}</strong>
            </label>
          </div>

          <div className="action-grid">
            <button onClick={applyParams} disabled={pending}>Apply Params</button>
            <button onClick={() => runAction(() => request("/api/state/evolve", { method: "POST", body: JSON.stringify({ dt: 0.2, steps: 1 }) }))} disabled={pending}>
              Evolve 1 Step
            </button>
            <button onClick={() => runAction(() => request("/api/state/evolve", { method: "POST", body: JSON.stringify({ dt: 0.1, steps: 5 }) }))} disabled={pending}>
              Evolve 5 Steps
            </button>
            <button onClick={() => runAction(() => request("/api/state/create", { method: "POST", body: JSON.stringify({ Lx: 2, Ly: 2, ...params }) }))} disabled={pending}>
              Recreate 2x2
            </button>
          </div>
        </article>

        <article className="panel phase-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Phase Output</p>
              <h2>Phase classifier</h2>
            </div>
          </div>

          <div className="source-stack">
            <span className={`source-pill ${modelStatus?.model_loaded ? "source-live" : "source-fallback"}`}>
              {modelStatus?.model_loaded ? "Trained model" : "Fallback rules"}
            </span>
            <p className="panel-note">
              Source {modelStatus?.source ?? "unknown"} · Artifact{" "}
              <code>{modelStatus?.model_path ?? "unavailable"}</code>
            </p>
          </div>

          <div className="phase-badge" style={{ borderColor: phaseColor, color: phaseColor }}>
            {state.phase.label}
          </div>
          <p className="phase-confidence">
            Confidence {(state.phase.confidence * 100).toFixed(1)}%
          </p>

          <div className="probability-list">
            {Object.entries(state.phase.probabilities).map(([label, probability]) => (
              <div key={label} className="probability-row">
                <span>{label}</span>
                <div className="probability-bar">
                  <div style={{ width: `${probability * 100}%`, background: PHASE_COLORS[label] ?? "#7d87a7" }} />
                </div>
                <strong>{(probability * 100).toFixed(0)}%</strong>
              </div>
            ))}
          </div>

          <p className="panel-note">
            The browser does not infer anything locally. It only displays the
            backend classifier result and the backend-reported provenance.
          </p>
        </article>

        <article className="panel metrics-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Benchmarks</p>
              <h2>Saved training metrics</h2>
            </div>
          </div>

          <div className="metric-grid">
            <div className="metric-card">
              <span>Metrics</span>
              <strong>{metrics?.available ? "Loaded" : "Missing"}</strong>
            </div>
            <div className="metric-card">
              <span>Val</span>
              <strong>{metrics?.val_accuracy != null ? `${(metrics.val_accuracy * 100).toFixed(1)}%` : "N/A"}</strong>
            </div>
            <div className="metric-card">
              <span>Test</span>
              <strong>{metrics?.test_accuracy != null ? `${(metrics.test_accuracy * 100).toFixed(1)}%` : "N/A"}</strong>
            </div>
            <div className="metric-card">
              <span>2x2→2x3</span>
              <strong>
                {metrics?.cross_lattice_accuracy != null
                  ? `${(metrics.cross_lattice_accuracy * 100).toFixed(1)}%`
                  : "N/A"}
              </strong>
            </div>
          </div>

          <p className="panel-note">
            Metrics file <code>{metrics?.metrics_path ?? "unavailable"}</code>
          </p>
        </article>

        <article className="panel observables-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Observables</p>
              <h2>Backend-derived signals</h2>
            </div>
          </div>

          <div className="metric-grid">
            <div className="metric-card">
              <span>D</span>
              <strong>{state.observables.D.toFixed(3)}</strong>
            </div>
            <div className="metric-card">
              <span>n</span>
              <strong>{state.observables.n.toFixed(3)}</strong>
            </div>
            <div className="metric-card">
              <span>Ms²</span>
              <strong>{state.observables.Ms2.toFixed(3)}</strong>
            </div>
            <div className="metric-card">
              <span>K</span>
              <strong>{state.observables.K.toFixed(3)}</strong>
            </div>
            <div className="metric-card">
              <span>Cs,max</span>
              <strong>{state.observables.Cs_max.toFixed(3)}</strong>
            </div>
            <div className="metric-card">
              <span>Energy</span>
              <strong>{state.observables.energy.toFixed(3)}</strong>
            </div>
          </div>

          <div className="bond-list">
            {state.lattice.bonds.map((bond) => (
              <div key={`${bond.i}-${bond.j}`} className="bond-row">
                <span>
                  Bond {bond.i}-{bond.j}
                </span>
                <strong>{bond.strength.toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}

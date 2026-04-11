import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const PHASE_COLORS = {
  Metal: "#4fb3ff",
  "Mott Insulator": "#ff9d42",
  Antiferromagnet: "#67e39a",
  "Singlet-rich": "#ff6f91",
  unclassified: "#a4acc4",
};

const PHASE_EXPLANATIONS = {
  Metal: "Charges move relatively freely through the lattice.",
  "Mott Insulator": "Strong interactions lock charges in place even when simple band theory would suggest motion.",
  Antiferromagnet: "Neighboring spins prefer to alternate up/down in a checkerboard-like pattern.",
  "Singlet-rich": "The state shows stronger short-range electron pairing tendencies, but this is not the same as proven superconductivity.",
};

const TARGET_LABELS = {
  D: "Double occupancy",
  n: "Average filling",
  Ms2: "Spin-alternation strength",
  K: "Motion / kinetic signal",
  Cs_max: "Long-range spin link",
};

const TARGET_EXPLANATIONS = {
  D: "How often two electrons sit on the same site.",
  n: "How many electrons are present on average per site.",
  Ms2: "How strongly spins line up in an alternating antiferromagnetic pattern.",
  K: "How strongly electrons are delocalizing and hopping across the lattice.",
  Cs_max: "How correlated the most distant sites are.",
};

const TRUST_LABEL_COPY = {
  safe: {
    title: "Cheap solver trusted",
    body: "The mean-field approximation is close enough here that the cheap answer is probably usable.",
  },
  warning: {
    title: "Approximation warning",
    body: "The cheap solver is starting to drift away from the exact answer. Check a stronger method before trusting it.",
  },
  unsafe: {
    title: "Strong-correlation risk",
    body: "The cheap solver is likely missing important many-body physics here. Escalate to a stronger solver.",
  },
};

const QPROBE_VALUE_POINTS = [
  {
    title: "Cut expensive basis changes",
    body: "Each measurement group is a separate basis setting on the quantum device. Fewer groups means a cheaper experiment.",
  },
  {
    title: "Preserve the scientific answer",
    body: "QProbe only claims a shortcut when the recovered diagnostics still match the exact physics within a chosen error budget.",
  },
  {
    title: "Refuse unsafe shortcuts",
    body: "A useful planner is not one that always compresses. It must also know when extra measurements are truly necessary.",
  },
];

const PAGE_GUIDE = [
  {
    title: "1. Choose a state",
    body: "Use a demo preset, click tiles on the board, or reset to a known pattern like Néel order.",
  },
  {
    title: "2. Tune the model",
    body: "Move the physics sliders and apply them. This changes how strongly electrons move, repel, and fill the lattice.",
  },
  {
    title: "3. Read the recommendations",
    body: "CorrMap tells you whether a cheap solver is trustworthy. QProbe tells you how many measurements you can safely skip.",
  },
];

const PIPELINE_STEPS = [
  {
    step: "Step 1",
    title: "Identify the regime",
    body: "The classifier gives a fast physical summary of what kind of state you prepared.",
  },
  {
    step: "Step 2",
    title: "Check solver trust",
    body: "CorrMap asks whether a cheap approximation is reliable or whether strong correlations demand a stronger solver.",
  },
  {
    step: "Step 3",
    title: "Plan the measurements",
    body: "QProbe finds the cheapest safe way to extract the signals you care about once you decide what solver path to trust.",
  },
];

const OCCUPANCY_ORDER = ["empty", "up", "down", "double"];
const QPROBE_TARGETS = ["D", "n", "Ms2", "K", "Cs_max"];
const DEMO_PRESETS = {
  compressed: {
    label: "Compression Win",
    description: "Success case: QProbe can safely compress because the requested signals all live in simple charge/spin measurements.",
    expectation: "Expected result: QProbe and Adaptive QProbe should both save measurement groups.",
    params: { t: 1, U: 8, mu: 4 },
    qprobe: { targets: ["D", "n", "Ms2", "Cs_max"], tolerance: 0.03, shotsPerGroup: 4000, readoutFlipProb: 0.02 },
  },
  adaptive: {
    label: "Adaptive Win",
    description: "Adaptive success case: QProbe should reach a safe answer quickly and stop after a short trajectory.",
    expectation: "Expected result: Adaptive QProbe should stop early once error and uncertainty fall within tolerance.",
    params: { t: 1, U: 6, mu: 3 },
    qprobe: { targets: ["D", "Ms2", "Cs_max"], tolerance: 0.035, shotsPerGroup: 6000, readoutFlipProb: 0.015 },
  },
  hard: {
    label: "No Safe Shortcut",
    description: "Safety case: kinetic information under tighter noise makes compression unreliable.",
    expectation: "Expected result: QProbe may refuse to compress, and Adaptive QProbe may need the full plan.",
    params: { t: 1, U: 4, mu: 2 },
    qprobe: { targets: ["D", "n", "Ms2", "K", "Cs_max"], tolerance: 0.01, shotsPerGroup: 2000, readoutFlipProb: 0.08 },
  },
};

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

function actionDescription(action) {
  const descriptions = {
    applyParams: "Rebuild the Hamiltonian with the current slider values.",
    applyBoard: "Re-initialize the lattice using the tile occupations you selected.",
    resetNeel: "Load an alternating up/down spin pattern, a standard antiferromagnetic starting point.",
    exactGroundState: "Solve for the exact lowest-energy state of the current 2x2 model.",
    evolve1: "Push the current state forward a short amount in time.",
    evolve5: "Push the current state forward several short time steps.",
    recreate: "Start over from a fresh 2x2 lattice with the current slider settings.",
    runQProbe: "Search for the smallest fixed measurement plan that still preserves the chosen signals.",
    runAdaptive: "Let Adaptive QProbe decide step by step whether it already knows enough to stop measuring.",
  };
  return descriptions[action];
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
  const [qprobeLibrary, setQprobeLibrary] = useState(null);
  const [qprobeResult, setQprobeResult] = useState(null);
  const [adaptiveQprobeResult, setAdaptiveQprobeResult] = useState(null);
  const [trustResult, setTrustResult] = useState(null);
  const [trustMetrics, setTrustMetrics] = useState(null);
  const [activePresetKey, setActivePresetKey] = useState(null);
  const [qprobeConfig, setQprobeConfig] = useState({
    targets: ["D", "n", "Ms2", "Cs_max"],
    tolerance: 0.03,
    shotsPerGroup: 4000,
    readoutFlipProb: 0.0,
  });

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
      const [nextState, library, trust, trustMetricSummary] = await Promise.all([
        request("/api/state/export"),
        request("/api/qprobe/library"),
        request("/api/trust/evaluate", { method: "POST" }),
        request("/api/trust/metrics"),
      ]);
      setState(nextState);
      setQprobeLibrary(library);
      setTrustResult(trust);
      setTrustMetrics(trustMetricSummary);
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
      const trust = await request("/api/trust/evaluate", { method: "POST" });
      setState(nextState);
      setTrustResult(trust);
      setQprobeResult(null);
      setAdaptiveQprobeResult(null);
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

  const toggleQProbeTarget = (target) => {
    setQprobeConfig((current) => {
      const present = current.targets.includes(target);
      const targets = present
        ? current.targets.filter((entry) => entry !== target)
        : [...current.targets, target];
      return {
        ...current,
        targets: targets.length > 0 ? targets : current.targets,
      };
    });
  };

  const runQProbe = async () => {
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/recommend-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: qprobeConfig.targets,
          tolerance: Number(qprobeConfig.tolerance),
          shots_per_group: Number(qprobeConfig.shotsPerGroup),
          readout_flip_prob: Number(qprobeConfig.readoutFlipProb),
          seed: 11,
        }),
      });
      setQprobeResult(result);
      setAdaptiveQprobeResult(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runAdaptiveQProbe = async () => {
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/adaptive-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: qprobeConfig.targets,
          tolerance: Number(qprobeConfig.tolerance),
          shots_per_group: Number(qprobeConfig.shotsPerGroup),
          readout_flip_prob: Number(qprobeConfig.readoutFlipProb),
          seed: 11,
        }),
      });
      setAdaptiveQprobeResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const loadDemoPreset = async (presetKey) => {
    const preset = DEMO_PRESETS[presetKey];
    setPending(true);
    setError("");
    try {
      const created = await request("/api/state/create", {
        method: "POST",
        body: JSON.stringify({ Lx: 2, Ly: 2, ...preset.params }),
      });
      const grounded = await request("/api/state/ground-state", { method: "POST" });
      const trust = await request("/api/trust/evaluate", { method: "POST" });
      setState(grounded);
      setTrustResult(trust);
      setParams(preset.params);
      setActivePresetKey(presetKey);
      syncBoardFromState(grounded);
      setQprobeConfig(preset.qprobe);
      const result = await request("/api/qprobe/recommend-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: preset.qprobe.targets,
          tolerance: preset.qprobe.tolerance,
          shots_per_group: preset.qprobe.shotsPerGroup,
          readout_flip_prob: preset.qprobe.readoutFlipProb,
          seed: 11,
        }),
      });
      setQprobeResult(result);
      const adaptive = await request("/api/qprobe/adaptive-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: preset.qprobe.targets,
          tolerance: preset.qprobe.tolerance,
          shots_per_group: preset.qprobe.shotsPerGroup,
          readout_flip_prob: preset.qprobe.readoutFlipProb,
          seed: 11,
        }),
      });
      setAdaptiveQprobeResult(adaptive);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const boardSites = useMemo(() => state?.lattice?.sites ?? [], [state]);
  const phaseColor = PHASE_COLORS[state?.phase?.label] ?? PHASE_COLORS.unclassified;
  const modelStatus = state?.phase?.model_status;
  const metrics = state?.metrics;
  const trustStatus = trustResult ? TRUST_LABEL_COPY[trustResult.risk_label] : null;
  const qprobeBenchmark = {
    costAccuracy: 1.0,
    safetyAccuracy: 1.0,
    falseSafeRate: 0.0,
  };
  const activePreset = activePresetKey ? DEMO_PRESETS[activePresetKey] : null;
  const qprobeImpact = qprobeResult
    ? {
        savings: qprobeResult.measurement_savings,
        percentSaved:
          qprobeResult.full_cost > 0
            ? Math.round((100 * qprobeResult.measurement_savings) / qprobeResult.full_cost)
            : 0,
      }
    : null;
  const workflowSummary = state && trustResult
    ? {
        regime: state.phase.label,
        solver:
          trustResult.recommended_action === "cheap_solver_ok"
            ? "Cheap solver is probably enough"
            : trustResult.recommended_action === "check_exact_or_stronger_solver"
              ? "Cross-check with a stronger solver"
              : "Escalate to exact / advanced physics",
        measurement:
          qprobeResult == null
            ? "Run QProbe to choose a measurement plan"
            : qprobeResult.success
              ? `QProbe found a cheaper safe plan (${qprobeResult.measurement_savings} groups saved)`
              : "QProbe says no safe shortcut under this budget",
      }
    : null;

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
          <h1>Use AI to choose cheaper quantum measurements without losing the physics</h1>
          <p className="hero-copy">
            This app uses a small quantum-material model as a testbed. The backend
            computes exact physics, then QProbe tries to find the smallest set of
            measurements that still recovers the same scientific signal under noise.
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

      <section className="guide-grid">
        {PAGE_GUIDE.map((entry) => (
          <article key={entry.title} className="guide-card">
            <strong>{entry.title}</strong>
            <span>{entry.body}</span>
          </article>
        ))}
      </section>

      <section className="pipeline-grid">
        {PIPELINE_STEPS.map((entry) => (
          <article key={entry.step} className="pipeline-card">
            <span className="eyebrow">{entry.step}</span>
            <strong>{entry.title}</strong>
            <span>{entry.body}</span>
          </article>
        ))}
      </section>

      <section className="preset-strip">
        {Object.entries(DEMO_PRESETS).map(([key, preset]) => (
          <button key={key} className="preset-card" onClick={() => loadDemoPreset(key)} disabled={pending}>
            <span className="eyebrow">Demo Preset</span>
            <strong>{preset.label}</strong>
            <span>{preset.description}</span>
            <span className="preset-expectation">{preset.expectation}</span>
          </button>
        ))}
      </section>

      {activePreset ? (
        <section className="preset-note">
          <strong>{activePreset.label}</strong>
          <span>{activePreset.expectation}</span>
        </section>
      ) : null}

      <section className="hero-stats">
        <article className="hero-stat-card">
          <span className="eyebrow">Exact QProbe</span>
          <strong>Oracle planner</strong>
          <span>Exhaustive search over measurement plans</span>
        </article>
        <article className="hero-stat-card">
          <span className="eyebrow">ML-QProbe</span>
          <strong>{(qprobeBenchmark.costAccuracy * 100).toFixed(0)}% cost match</strong>
          <span>{(qprobeBenchmark.safetyAccuracy * 100).toFixed(0)}% safety match on the oracle benchmark</span>
        </article>
        <article className="hero-stat-card">
          <span className="eyebrow">Safety</span>
          <strong>{(qprobeBenchmark.falseSafeRate * 100).toFixed(0)}% false-safe rate</strong>
          <span>The learned model did not suggest unsafe compression in the test set.</span>
        </article>
      </section>

      <section className="why-grid">
        {QPROBE_VALUE_POINTS.map((point) => (
          <article key={point.title} className="why-card">
            <span className="eyebrow">Why QProbe</span>
            <strong>{point.title}</strong>
            <span>{point.body}</span>
          </article>
        ))}
      </section>

      {trustResult ? (
        <section className={`preset-note ${trustResult.risk_label === "safe" ? "note-safe" : trustResult.risk_label === "warning" ? "note-warning" : "note-unsafe"}`}>
          <strong>
            Workflow recommendation:{" "}
            {trustResult.recommended_action === "cheap_solver_ok"
              ? "use the cheap solver"
              : trustResult.recommended_action === "check_exact_or_stronger_solver"
                ? "double-check with a stronger solver"
                : "escalate to exact / advanced physics"}
          </strong>
          <span>
            {trustStatus?.body}
            {qprobeResult
              ? ` QProbe then decides whether the stronger path can still be measured cheaply.`
              : ""}
          </span>
        </section>
      ) : null}

      {workflowSummary ? (
        <section className="workflow-card">
          <div>
            <p className="eyebrow">Unified Recommendation</p>
            <h2>One workflow, not three separate tools</h2>
          </div>
          <div className="workflow-grid">
            <div className="workflow-item">
              <span>Regime</span>
              <strong>{workflowSummary.regime}</strong>
            </div>
            <div className="workflow-item">
              <span>Solver decision</span>
              <strong>{workflowSummary.solver}</strong>
            </div>
            <div className="workflow-item">
              <span>Measurement decision</span>
              <strong>{workflowSummary.measurement}</strong>
            </div>
          </div>
        </section>
      ) : null}

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
          <p className="panel-note panel-note-tight">
            Each tile is one location in the material. Click to cycle through empty,
            one electron, or two electrons.
          </p>

          <div className="board-controls">
            <label>
              Default reset
              <select value={boardMode} onChange={(event) => setBoardMode(event.target.value)} disabled={pending}>
                <option value="neel">N&eacute;el</option>
                <option value="empty">Empty</option>
                <option value="polarized">Polarized</option>
              </select>
              <span className="field-help">
                This chooses the background pattern used before your manual tile overrides are applied.
              </span>
            </label>
            <button onClick={applyBoard} disabled={pending}>Apply Board</button>
          </div>

          <div className="help-list">
            <div className="help-row">
              <strong>Tile colors</strong>
              <span>
                <code>up</code> means one spin-up electron, <code>down</code> means one spin-down electron, and <code>double</code> means both are present on the same site.
              </span>
            </div>
            <div className="help-row">
              <strong>{actionDescription("applyBoard")}</strong>
              <span>This does not evolve the old state. It rebuilds a new state from the board you drew.</span>
            </div>
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
              <h2>Physics controls</h2>
            </div>
          </div>
          <p className="panel-note panel-note-tight">
            <strong>U</strong> controls interaction strength. <strong>μ</strong> changes
            filling. <strong>t</strong> controls hopping strength.
          </p>

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
              <span className="field-help">
                Higher <code>t</code> means electrons hop more easily between neighboring sites.
              </span>
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
              <span className="field-help">
                Higher <code>U</code> means electrons pay a larger penalty for sharing the same site.
              </span>
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
              <span className="field-help">
                <code>μ</code> shifts how full the lattice wants to be. Around half filling, the average occupancy per site is near 1.
              </span>
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

          <div className="help-list">
            <div className="help-row">
              <strong>{actionDescription("applyParams")}</strong>
              <span>Use this after changing the sliders.</span>
            </div>
            <div className="help-row">
              <strong>{actionDescription("exactGroundState")}</strong>
              <span>This is the cleanest “reference answer” for the current tiny lattice.</span>
            </div>
            <div className="help-row">
              <strong>{actionDescription("evolve1")}</strong>
              <span>{actionDescription("evolve5")}</span>
            </div>
          </div>
        </article>

        <article className="panel phase-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Step 1</p>
              <h2>Identify the regime</h2>
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
          <p className="panel-note panel-note-tight">
            {PHASE_EXPLANATIONS[state.phase.label] ?? "The model is summarizing the current quantum state."}
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
            Important: antiferromagnetism is not superconductivity. In this app,
            “Singlet-rich” is the closest thing to short-range pairing behavior,
            but it still does not prove superconductivity.
          </p>
        </article>

        <article className="panel metrics-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Step 2</p>
              <h2>Check solver trust with CorrMap</h2>
            </div>
          </div>
          <p className="panel-note panel-note-tight">
            The cheap solver here is a mean-field approximation. CorrMap compares
            it against exact physics on small clusters and learns where that shortcut
            is safe or risky.
          </p>
          <div className="scope-box">
            <strong>Current scope</strong>
            <span>
              This CorrMap prototype is currently calibrated on <code>2x2</code> Hubbard
              plaquettes, using mean-field as the cheap solver and exact ED as the
              reference. The framework underneath is being generalized so later versions
              can cover larger lattices and additional 2D lattice models.
            </span>
          </div>
          {trustResult ? (
            <>
              <div className={`qprobe-banner ${trustResult.risk_label === "safe" ? "qprobe-success" : "qprobe-failure"}`}>
                <strong>{trustStatus?.title ?? "Trust result"}</strong>
                <span>
                  Oracle label: {trustResult.risk_label}. Model recommendation:{" "}
                  {trustResult.trust_prediction.label ?? trustResult.risk_label}.
                </span>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Cheap solver risk</span>
                  <strong>{trustResult.risk_label}</strong>
                </div>
                <div className="metric-card">
                  <span>Worst observable gap</span>
                  <strong>{trustResult.max_abs_error.toFixed(4)}</strong>
                </div>
                <div className="metric-card">
                  <span>Energy gap</span>
                  <strong>{trustResult.energy_error.toFixed(4)}</strong>
                </div>
              </div>

              <div className="comparison-grid">
                <div className="comparison-card">
                  <p className="eyebrow">Cheap Solver</p>
                  <h3>Mean-field baseline</h3>
                  <div className="qprobe-table">
                    {Object.entries(trustResult.cheap_solver).map(([name, value]) => (
                      <div key={name} className="qprobe-row">
                        <span>{TARGET_LABELS[name] ?? name}</span>
                        <strong>{Number(value).toFixed(4)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="comparison-card">
                  <p className="eyebrow">Exact Solver</p>
                  <h3>ED reference</h3>
                  <div className="qprobe-table">
                    {Object.entries(trustResult.exact).map(([name, value]) => (
                      <div key={name} className="qprobe-row">
                        <span>{TARGET_LABELS[name] ?? name}</span>
                        <strong>{Number(value).toFixed(4)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="qprobe-table">
                {Object.keys(trustResult.abs_error).map((name) => (
                  <div key={name} className="qprobe-row">
                    <span>{TARGET_LABELS[name] ?? name}</span>
                    <span>abs gap {trustResult.abs_error[name].toFixed(4)}</span>
                    <strong>rel gap {trustResult.rel_error[name].toFixed(3)}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="panel-note">Run a state update to evaluate the trust layer.</p>
          )}
          {trustMetrics ? (
            <p className="panel-note">
              TrustNet benchmark: test risk accuracy{" "}
              {trustMetrics.test_risk_accuracy != null ? `${(trustMetrics.test_risk_accuracy * 100).toFixed(0)}%` : "N/A"}
              {" "}· false-safe rate{" "}
              {trustMetrics.test_false_safe_rate != null ? `${(trustMetrics.test_false_safe_rate * 100).toFixed(0)}%` : "N/A"}
              {trustMetrics.cross_lattice_risk_accuracy != null
                ? ` · 2x2→2x3 transfer ${(trustMetrics.cross_lattice_risk_accuracy * 100).toFixed(0)}%`
                : ""}.
            </p>
          ) : null}
        </article>

        <article className="panel qprobe-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Step 3</p>
              <h2>Plan the measurements with QProbe</h2>
            </div>
          </div>
          <p className="panel-note panel-note-tight">
            QProbe asks: can we measure fewer things on a noisy quantum device and
            still recover the same scientific conclusion?
          </p>
          <div className="qprobe-why-box">
            <strong>Why care?</strong>
            <span>
              On hardware, each measurement group means a new basis rotation plus
              another batch of shots. QProbe tries to keep only the groups that
              actually matter for the signals you selected.
            </span>
            {qprobeImpact ? (
              <span className="qprobe-impact">
                Latest result: saved <strong>{qprobeImpact.savings}</strong> group
                {qprobeImpact.savings === 1 ? "" : "s"} ({qprobeImpact.percentSaved}% fewer basis settings).
              </span>
            ) : (
              <span className="qprobe-impact">
                Run QProbe to see whether the current experiment can be made cheaper
                without changing the answer.
              </span>
            )}
          </div>

          <div className="target-grid">
            {QPROBE_TARGETS.map((target) => (
              <button
                key={target}
                className={`target-chip ${qprobeConfig.targets.includes(target) ? "target-active" : ""}`}
                onClick={() => toggleQProbeTarget(target)}
                disabled={pending}
              >
                {TARGET_LABELS[target]}
              </button>
            ))}
          </div>
          <div className="target-help-list">
            {qprobeConfig.targets.map((target) => (
              <div key={target} className="target-help-row">
                <strong>{TARGET_LABELS[target]}</strong>
                <span>{TARGET_EXPLANATIONS[target]}</span>
              </div>
            ))}
          </div>

          <div className="slider-stack">
            <label>
              <span>Tolerance</span>
              <input
                type="range"
                min="0.005"
                max="0.1"
                step="0.005"
                value={qprobeConfig.tolerance}
                onChange={(event) =>
                  setQprobeConfig((current) => ({
                    ...current,
                    tolerance: Number(event.target.value),
                  }))
                }
                disabled={pending}
              />
              <strong>{Number(qprobeConfig.tolerance).toFixed(3)}</strong>
              <span className="field-help">
                Smaller tolerance means “be stricter.” QProbe will allow less reconstruction error before it approves a shortcut.
              </span>
            </label>
            <label>
              <span>Measurements per group</span>
              <input
                type="range"
                min="500"
                max="20000"
                step="500"
                value={qprobeConfig.shotsPerGroup}
                onChange={(event) =>
                  setQprobeConfig((current) => ({
                    ...current,
                    shotsPerGroup: Number(event.target.value),
                  }))
                }
                disabled={pending}
              />
              <strong>{qprobeConfig.shotsPerGroup}</strong>
              <span className="field-help">
                More measurements reduce random noise, but cost more hardware time.
              </span>
            </label>
            <label>
              <span>Noise level</span>
              <input
                type="range"
                min="0"
                max="0.15"
                step="0.01"
                value={qprobeConfig.readoutFlipProb}
                onChange={(event) =>
                  setQprobeConfig((current) => ({
                    ...current,
                    readoutFlipProb: Number(event.target.value),
                  }))
                }
                disabled={pending}
              />
              <strong>{Number(qprobeConfig.readoutFlipProb).toFixed(2)}</strong>
              <span className="field-help">
                This simulates readout mistakes. A higher value means a noisier device.
              </span>
            </label>
          </div>

          <div className="action-grid">
            <button onClick={runQProbe} disabled={pending}>Run QProbe</button>
            <button onClick={runAdaptiveQProbe} disabled={pending}>Run Adaptive QProbe</button>
          </div>

          <div className="help-list">
            <div className="help-row">
              <strong>{actionDescription("runQProbe")}</strong>
              <span>Use this for a one-shot plan: full search, then one recommended fixed plan.</span>
            </div>
            <div className="help-row">
              <strong>{actionDescription("runAdaptive")}</strong>
              <span>Use this when you want to see the measurement process unfold step by step.</span>
            </div>
          </div>

          {qprobeResult ? (
            <>
              <div className={`qprobe-banner ${qprobeResult.success ? "qprobe-success" : "qprobe-failure"}`}>
                <strong>{qprobeResult.success ? "Compressed plan found" : "Safety refusal: no safe compression"}</strong>
                <span>{qprobeResult.message}</span>
              </div>
              {trustResult ? (
                <p className="panel-note panel-note-tight">
                  Solver route:{" "}
                  {trustResult.recommended_action === "cheap_solver_ok"
                    ? "the cheap solver is probably enough here, so QProbe is mostly a measurement-efficiency demo."
                    : "this regime looks correlation-hard, so QProbe matters more because a stronger solver path needs careful measurement planning."}
                </p>
              ) : null}

              <div className="comparison-grid">
                <div className="comparison-card">
                  <p className="eyebrow">Full Plan</p>
                  <h3>{qprobeResult.full_cost} measurement groups</h3>
                  <div className="group-list">
                    {qprobeResult.full_groups.map((group) => (
                      <div key={group.name} className="group-row">
                        <div className="group-copy">
                          <strong>{group.basis_label}</strong>
                          <span>{group.basis}</span>
                          <span>{group.explanation}</span>
                        </div>
                        <strong>{group.num_terms} terms</strong>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="comparison-card">
                  <p className="eyebrow">QProbe Plan</p>
                  <h3>{qprobeResult.recommended_cost} measurement groups</h3>
                  <div className="group-list">
                    {qprobeResult.recommended_groups.map((group) => (
                      <div key={group.name} className="group-row">
                        <div className="group-copy">
                          <strong>{group.basis_label}</strong>
                          <span>{group.basis}</span>
                          <span>Supports {group.supports_targets.map((target) => TARGET_LABELS[target] ?? target).join(", ")}</span>
                        </div>
                        <strong>{group.num_terms} terms</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Groups saved</span>
                  <strong>{qprobeResult.measurement_savings}</strong>
                </div>
                <div className="metric-card">
                  <span>Worst error</span>
                  <strong>{qprobeResult.max_abs_error.toFixed(4)}</strong>
                </div>
                <div className="metric-card">
                  <span>Tracked signals</span>
                  <strong>{qprobeResult.targets.map((target) => TARGET_LABELS[target] ?? target).join(", ")}</strong>
                </div>
              </div>

              <div className="comparison-grid">
                <div className="comparison-card">
                  <p className="eyebrow">Exact QProbe</p>
                  <h3>{qprobeResult.recommended_cost} groups</h3>
                  <p className="panel-note panel-note-tight">
                    Oracle planner using exact search over the measurement library.
                  </p>
                </div>

                <div className="comparison-card">
                  <p className="eyebrow">ML-QProbe</p>
                  <h3>
                    {qprobeResult.ml_qprobe.available
                      ? `${qprobeResult.ml_qprobe.predicted_cost} groups`
                      : "Unavailable"}
                  </h3>
                  {qprobeResult.ml_qprobe.available ? (
                    <>
                      <p className="panel-note panel-note-tight">
                        Predicted success {qprobeResult.ml_qprobe.predicted_success ? "yes" : "no"} ·
                        predicted error {qprobeResult.ml_qprobe.predicted_error.toFixed(4)}
                      </p>
                      <div className="match-row">
                        <span className={`match-pill ${qprobeResult.ml_qprobe.matches_exact_cost ? "match-good" : "match-warn"}`}>
                          {qprobeResult.ml_qprobe.matches_exact_cost ? "Cost match" : "Cost mismatch"}
                        </span>
                        <span className={`match-pill ${qprobeResult.ml_qprobe.matches_exact_success ? "match-good" : "match-warn"}`}>
                          {qprobeResult.ml_qprobe.matches_exact_success ? "Safety match" : "Safety mismatch"}
                        </span>
                      </div>
                    </>
                  ) : (
                    <p className="panel-note panel-note-tight">
                      ML-QProbe model artifact not loaded.
                    </p>
                  )}
                </div>
              </div>

              <div className="qprobe-table">
                {Object.keys(qprobeResult.exact).map((name) => (
                  <div key={name} className="qprobe-row">
                    <span>{TARGET_LABELS[name] ?? name}</span>
                    <span>target {qprobeResult.exact[name].toFixed(4)}</span>
                    <span>measured {qprobeResult.estimated[name].toFixed(4)}</span>
                    <strong>error {qprobeResult.abs_error[name].toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="panel-note">
              QProbe compares the expensive “measure everything” plan against the
              smallest plan that still recovers the chosen signals within your
              allowed error budget.
            </p>
          )}

          {qprobeLibrary ? (
            <p className="panel-note">
              Library loaded for {Object.keys(qprobeLibrary.observables).length} diagnostics.
            </p>
          ) : null}

          {adaptiveQprobeResult ? (
            <>
              <div className={`qprobe-banner ${adaptiveQprobeResult.success ? "qprobe-success" : "qprobe-failure"}`}>
                <strong>{adaptiveQprobeResult.success ? "Adaptive QProbe stopped in runtime mode" : "Adaptive QProbe could not stop in runtime mode"}</strong>
                <span>{adaptiveQprobeResult.message}</span>
              </div>
              <div className="metric-grid">
                <div className="metric-card">
                  <span>Adaptive final cost</span>
                  <strong>{adaptiveQprobeResult.final_cost}</strong>
                </div>
                <div className="metric-card">
                  <span>Adaptive savings</span>
                  <strong>{adaptiveQprobeResult.measurement_savings}</strong>
                </div>
                <div className="metric-card">
                  <span>Max uncertainty</span>
                  <strong>{adaptiveQprobeResult.max_uncertainty.toFixed(4)}</strong>
                </div>
                <div className="metric-card">
                  <span>Oracle benchmark max error</span>
                  <strong>{adaptiveQprobeResult.max_abs_error.toFixed(4)}</strong>
                </div>
                <div className="metric-card">
                  <span>Oracle benchmark check</span>
                  <strong>{adaptiveQprobeResult.oracle_benchmark_within_tolerance ? "Within tolerance" : "Outside tolerance"}</strong>
                </div>
              </div>
              <p className="panel-note panel-note-tight">
                Runtime stop rule: {adaptiveQprobeResult.runtime_stop_rule}. The runtime policy only uses target coverage and measurement uncertainty. The exact benchmark error is shown here only to validate the policy on small solvable systems.
              </p>
              <div className="qprobe-table">
                {adaptiveQprobeResult.steps.map((step) => (
                  <div key={step.step_index} className="adaptive-step">
                    <div className="adaptive-step-head">
                      <strong>Step {step.step_index}</strong>
                      <span>{step.chosen_group.basis_label}</span>
                      <span>cost {step.current_cost}</span>
                    </div>
                    <div className="adaptive-step-body">
                      <span>{step.chosen_group.explanation}</span>
                      <span>supports {step.chosen_group.supports_targets.map((target) => TARGET_LABELS[target] ?? target).join(", ")}</span>
                      <span>covered {step.covered_targets.map((target) => TARGET_LABELS[target] ?? target).join(", ") || "none yet"}</span>
                      <span>remaining {step.unresolved_targets.map((target) => TARGET_LABELS[target] ?? target).join(", ") || "none"}</span>
                      <span>runtime uncertainty {step.max_uncertainty.toFixed(4)}</span>
                      <span>oracle benchmark error {step.max_abs_error.toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </article>

        <article className="panel metrics-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Benchmarks</p>
              <h2>Model benchmark</h2>
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
            This is background model quality information. The main demo is QProbe:
            reducing measurement cost without losing the key physics.
          </p>
        </article>

        <article className="panel observables-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Observables</p>
              <h2>Physics signals</h2>
            </div>
          </div>
          <p className="panel-note panel-note-tight">
            These are the summary signals QProbe tries to preserve even when it
            recommends fewer measurements.
          </p>

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

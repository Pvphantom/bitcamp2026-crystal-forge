import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const TARGET_LABELS = {
  D: "Double occupancy",
  n: "Average filling",
  Ms2: "Staggered spin order",
  K: "Transport / kinetic signal",
  Cs_max: "Long-range spin correlation",
  Pair_nn: "Nearest-neighbor pair density",
  Pair_span: "Long-range pair coherence",
};

const TARGET_GROUPS = {
  charge: ["n", "D"],
  spin: ["Ms2", "Cs_max"],
  transport: ["K"],
  pairing: ["Pair_nn", "Pair_span"],
};

function ChannelIcon({ channel }) {
  if (channel === "charge") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
        <circle cx="12" cy="12" r="1.8" fill="currentColor" />
      </svg>
    );
  }
  if (channel === "spin") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 6c2 0 3 1.6 3 3.2S9 12 7 12s-3-1.6-3-2.8S5 6 7 6Zm10 6c2 0 3 1.6 3 2.8S19 18 17 18s-3-1.6-3-2.8 1-3.2 3-3.2Z" fill="currentColor" />
        <path d="M8.5 10.5 15.5 13.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    );
  }
  if (channel === "transport") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 12h13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        <path d="m13 7 6 5-6 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 18s-6-3.8-6-8.3A3.7 3.7 0 0 1 12 7a3.7 3.7 0 0 1 6 2.7C18 14.2 12 18 12 18Z" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 18s-6-3.8-6-8.3A3.7 3.7 0 0 1 12 7a3.7 3.7 0 0 1 6 2.7C18 14.2 12 18 12 18Z" fill="currentColor" fillOpacity="0.16" />
    </svg>
  );
}

const DEFAULT_CONFIG = {
  modelFamily: "hubbard",
  Lx: 2,
  Ly: 2,
  parameters: { t: 1.0, U: 6.0, mu: 1.5 },
  qprobeTargets: ["n", "D", "Ms2"],
  qprobeTolerance: 0.05,
  qprobeShotsPerGroup: 2000,
  qprobeReadoutFlipProb: 0.02,
  qprobeSeed: 17,
};

const MAX_QPROBE_TARGETS = 5;
const EXACT_DEMO_TARGET_LIMIT = 3;
const LEGACY_QPROBE_TARGETS = ["n", "D", "Ms2", "K", "Cs_max"];
const GATE_SENSITIVE_TARGETS = ["D", "K", "Cs_max"];
const GATE_SENSITIVE_TOLERANCE = 0.15;

const BENCHMARK_SERIES = [
  {
    label: "Naive channelized",
    tolerance: "0.03",
    value: 21.9,
    tone: "baseline",
  },
  {
    label: "Decomposed workflow",
    tolerance: "0.03",
    value: 41.8,
    tone: "progress",
  },
  {
    label: "Decomposed workflow",
    tolerance: "0.05",
    value: 55.7,
    tone: "progress",
  },
  {
    label: "Decomposed workflow",
    tolerance: "0.08",
    value: 67.1,
    tone: "strong",
  },
];

const RECOVERABLE_SERIES = [
  { tolerance: "0.03", value: 61.3 },
  { tolerance: "0.05", value: 65.8 },
  { tolerance: "0.08", value: 65.8 },
];

const CORRMAP_RESULTS = [
  { label: "Overall", value: "85.0%" },
  { label: "Held-out 8x8", value: "86.7%" },
  { label: "Hubbard", value: "90.0%" },
];

const DEMO_PRESETS = {
  quantum: {
    label: "Quantum demo",
    config: {
      ...DEFAULT_CONFIG,
      parameters: { t: 0.8, U: 8.0, mu: 1.2 },
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    },
  },
  classical: {
    label: "Classical demo",
    config: {
      ...DEFAULT_CONFIG,
      parameters: { t: 1.0, U: 5.0, mu: 0.5 },
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    },
  },
  strongDrop: {
    label: "Strong gate drop",
    config: {
      ...DEFAULT_CONFIG,
      parameters: { t: 0.4, U: 4.0, mu: 0.5 },
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    },
  },
  noDrop: {
    label: "No gate drop",
    config: {
      ...DEFAULT_CONFIG,
      parameters: { t: 1.0, U: 6.0, mu: 1.5 },
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    },
  },
};

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

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(3) : value;
}

function basisRotationGateCost(basis) {
  return [...basis].reduce((total, symbol) => {
    if (symbol === "X") return total + 1;
    if (symbol === "Y") return total + 2;
    return total;
  }, 0);
}

function benchmarkBackedChannelStatus(channel, tolerance) {
  if (channel === "charge") {
    return {
      status: "Safe",
      tone: "safe",
      note: "Charge diagnostics are the most stable part of the workflow and are directly handled by adaptive planning.",
    };
  }
  if (channel === "spin") {
    if (tolerance >= 0.05) {
      return {
        status: "Safe",
        tone: "safe",
        note: "Spin diagnostics become directly reliable once tolerance is moderately relaxed.",
      };
    }
    return {
      status: "Recoverable",
      tone: "recoverable",
      note: "Spin is close to direct-safe and is often recoverable through the structured workflow.",
    };
  }
  if (channel === "transport") {
    if (tolerance >= 0.1) {
      return {
        status: "Recoverable",
        tone: "recoverable",
        note: "Transport remains hard; the workflow decomposes it, but it is still not a fully direct-safe channel.",
      };
    }
    return {
      status: "Unresolved",
      tone: "unresolved",
      note: "Transport is one of the hard superconductivity channels and still requires decomposition or future RDM-style methods.",
    };
  }
  if (tolerance >= 0.08) {
    return {
      status: "Recoverable",
      tone: "recoverable",
      note: "Pairing becomes mostly usable through decomposed subtargets at moderate tolerance, even when not every raw piece is direct-safe.",
    };
  }
  return {
    status: "Unresolved",
    tone: "unresolved",
    note: "Pairing is the key hard sector. The workflow improves it, but direct-safe coverage is still incomplete.",
  };
}

function workflowCoverageForTolerance(tolerance) {
  if (tolerance >= 0.08) {
    return { safe: 67.1, recoverable: 65.8 };
  }
  if (tolerance >= 0.05) {
    return { safe: 55.7, recoverable: 65.8 };
  }
  return { safe: 41.8, recoverable: 61.3 };
}

function selectedChannels(targets) {
  return Object.entries(TARGET_GROUPS)
    .filter(([, names]) => names.some((name) => targets.includes(name)))
    .map(([key]) => key);
}

function MetricRow({ label, value, hint }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

function familyKey(group) {
  return `${group.name}|${group.basis}`;
}

function ProgressBar({ label, value, tone, meta }) {
  return (
    <div className="progress-row">
      <div className="progress-header">
        <strong>{label}</strong>
        <span>{meta}</span>
      </div>
      <div className="progress-track">
        <div className={`progress-fill progress-${tone}`} style={{ width: `${value}%` }} />
      </div>
      <div className="progress-value">{value.toFixed(1)}%</div>
    </div>
  );
}

function MissionMap({ routeSummary, activeChannels }) {
  const escalated = Boolean(routeSummary?.escalation_triggered);
  return (
    <section className="panel mission-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Mission map</p>
          <h2>From routing to superconductivity signals</h2>
        </div>
      </div>
      <div className="mission-track">
        <div className="mission-node active">
          <span className="mission-node-label">Launch</span>
          <strong>Hubbard setup</strong>
        </div>
        <div className="mission-line active" />
        <div className={`mission-node ${routeSummary ? "active" : ""}`}>
          <span className="mission-node-label">Route</span>
          <strong>{escalated ? "Quantum frontier" : "Classical scalable"}</strong>
        </div>
        <div className={`mission-line ${activeChannels.length > 0 ? "active" : ""}`} />
        <div className={`mission-node ${activeChannels.length > 0 ? "active" : ""}`}>
          <span className="mission-node-label">Workflow</span>
          <strong>{activeChannels.length} active channels</strong>
        </div>
        <div className={`mission-line ${activeChannels.length > 0 ? "active" : ""}`} />
        <div className={`mission-node ${activeChannels.length > 0 ? "active" : ""}`}>
          <span className="mission-node-label">Outcome</span>
          <strong>Safe / recoverable map</strong>
        </div>
      </div>
      <div className="mission-orbit">
        <span className="mission-planet mission-planet-a" />
        <span className="mission-planet mission-planet-b" />
        <span className="mission-satellite" />
      </div>
    </section>
  );
}

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [workflowResult, setWorkflowResult] = useState(null);
  const [stateExport, setStateExport] = useState(null);
  const [routingResult, setRoutingResult] = useState(null);
  const [qprobeExact, setQprobeExact] = useState(null);
  const [qprobeAdaptive, setQprobeAdaptive] = useState(null);

  const tolerance = Number(config.qprobeTolerance);
  const selected = config.qprobeTargets;
  const activeChannels = useMemo(() => selectedChannels(selected), [selected]);
  const workflowCoverage = useMemo(() => workflowCoverageForTolerance(tolerance), [tolerance]);
  const qprobeExactCompatible =
    config.qprobeTargets.length > 0 &&
    config.qprobeTargets.length <= EXACT_DEMO_TARGET_LIMIT &&
    config.qprobeTargets.every((name) => LEGACY_QPROBE_TARGETS.includes(name));
  const qprobeAdaptiveCompatible = qprobeExactCompatible;
  const demoQprobeTargets = qprobeExactCompatible ? config.qprobeTargets : [];

  const loadState = async () => {
    const exported = await request("/api/state/export");
    setStateExport(exported);
    return exported;
  };

  const applyParamsToState = async () => {
    setPending(true);
    setError("");
    try {
      await request("/api/state/set-params", {
        method: "POST",
        body: JSON.stringify(config.parameters),
      });
      const grounded = await request("/api/state/ground-state", { method: "POST" });
      setStateExport(grounded);
      setWorkflowResult(null);
      setRoutingResult(null);
      setQprobeExact(null);
      setQprobeAdaptive(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runCorrMap = async () => {
    setPending(true);
    setError("");
    try {
      const result = await request("/api/routing/evaluate", {
        method: "POST",
        body: JSON.stringify({
          model_family: "hubbard",
          Lx: config.Lx,
          Ly: config.Ly,
          parameters: config.parameters,
          qprobe_targets: config.qprobeTargets,
          qprobe_tolerance: Number(config.qprobeTolerance),
          qprobe_shots_per_group: Number(config.qprobeShotsPerGroup),
          qprobe_readout_flip_prob: Number(config.qprobeReadoutFlipProb),
          qprobe_seed: Number(config.qprobeSeed),
        }),
      });
      setRoutingResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runWorkflow = async (nextConfig = config) => {
    setPending(true);
    setError("");
    try {
      const result = await request("/api/workflow/analyze", {
        method: "POST",
        body: JSON.stringify({
          model_family: "hubbard",
          Lx: nextConfig.Lx,
          Ly: nextConfig.Ly,
          parameters: nextConfig.parameters,
          qprobe_targets: nextConfig.qprobeTargets,
          qprobe_tolerance: Number(nextConfig.qprobeTolerance),
          qprobe_shots_per_group: Number(nextConfig.qprobeShotsPerGroup),
          qprobe_readout_flip_prob: Number(nextConfig.qprobeReadoutFlipProb),
          qprobe_seed: Number(nextConfig.qprobeSeed),
        }),
      });
      setWorkflowResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runExactQProbe = async () => {
    if (!qprobeExactCompatible) {
      setError(
        `Live QProbe demo supports up to ${EXACT_DEMO_TARGET_LIMIT} tractable legacy observables (${LEGACY_QPROBE_TARGETS.join(", ")}).`,
      );
      return;
    }
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/recommend-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: demoQprobeTargets,
          tolerance: Number(config.qprobeTolerance),
          shots_per_group: Number(config.qprobeShotsPerGroup),
          readout_flip_prob: Number(config.qprobeReadoutFlipProb),
          seed: Number(config.qprobeSeed),
        }),
      });
      setQprobeExact(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runAdaptiveQProbe = async () => {
    if (!qprobeAdaptiveCompatible) {
      setError(
        `Live QProbe demo supports up to ${EXACT_DEMO_TARGET_LIMIT} tractable legacy observables (${LEGACY_QPROBE_TARGETS.join(", ")}).`,
      );
      return;
    }
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/predict", {
        method: "POST",
        body: JSON.stringify({
          targets: demoQprobeTargets,
          tolerance: Number(config.qprobeTolerance),
          shots_per_group: Number(config.qprobeShotsPerGroup),
          readout_flip_prob: Number(config.qprobeReadoutFlipProb),
          seed: Number(config.qprobeSeed),
        }),
      });
      setQprobeAdaptive(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  useEffect(() => {
    loadState();
  }, []);

  const routeSummary = routingResult?.workflow_decision ?? workflowResult?.workflow_decision ?? null;
  const routingSummary = routingResult?.routing ?? workflowResult?.routing ?? null;
  const lattice = stateExport?.lattice ?? null;
  const phase = stateExport?.phase ?? null;

  const corrmapRouteLabel = routeSummary?.route_label ?? routingSummary?.route_label ?? null;
  const corrmapRouteDisplay =
    corrmapRouteLabel === "quantum_frontier"
      ? "Quantum frontier"
      : corrmapRouteLabel === "scalable_classical"
        ? "Classical scalable"
      : corrmapRouteLabel === "mean_field"
          ? "Classical scalable"
          : "Waiting for backend";
  const corrmapActionDisplay =
    corrmapRouteLabel === "quantum_frontier"
      ? "Escalate to quantum workflow and QProbe"
      : corrmapRouteLabel === "scalable_classical"
        ? "Use classical Monte Carlo sampling"
      : routeSummary?.recommendation ?? routingSummary?.recommended_action ?? "n/a";
  const corrmapIntrinsicDisplay =
    routingSummary?.intrinsic_label === "stable_classical"
      ? "Stable classical regime"
      : routingSummary?.intrinsic_label === "fragile_classical"
        ? "Fragile but still classical regime"
        : routingSummary?.intrinsic_label === "frontier_or_uncertain"
          ? "Frontier pressure / uncertainty"
          : "Binary Hubbard routing";
  const corrmapConfidenceScore =
    corrmapRouteLabel === "quantum_frontier"
      ? routingSummary?.candidate_scores?.quantum_frontier
      : corrmapRouteLabel === "scalable_classical"
        ? routingSummary?.candidate_scores?.scalable_classical
        : null;
  const corrmapConfidenceDisplay =
    typeof corrmapConfidenceScore === "number"
      ? `${(corrmapConfidenceScore * 100).toFixed(1)}%`
      : "n/a";
  const qprobeAllowed = corrmapRouteLabel === "quantum_frontier";
  const qprobeExactFullGateCost = qprobeExact
    ? qprobeExact.full_groups.reduce((sum, group) => sum + basisRotationGateCost(group.basis), 0)
    : null;
  const qprobeExactRecommendedGateCost = qprobeExact
    ? qprobeExact.recommended_groups.reduce((sum, group) => sum + basisRotationGateCost(group.basis), 0)
    : null;
  const qprobeExactGateSavings =
    qprobeExactFullGateCost != null && qprobeExactRecommendedGateCost != null
      ? qprobeExactFullGateCost - qprobeExactRecommendedGateCost
      : null;
  const qprobeCircuitReductionPct =
    qprobeExact && qprobeExact.full_cost > 0
      ? ((qprobeExact.full_cost - qprobeExact.recommended_cost) / qprobeExact.full_cost) * 100
      : null;
  const qprobeGateReductionPct =
    qprobeExactFullGateCost != null && qprobeExactFullGateCost > 0 && qprobeExactRecommendedGateCost != null
      ? ((qprobeExactFullGateCost - qprobeExactRecommendedGateCost) / qprobeExactFullGateCost) * 100
      : null;

  const updateParam = (key, value) => {
    setConfig((current) => ({
      ...current,
      parameters: {
        ...current.parameters,
        [key]: Number(value),
      },
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    }));
    setRoutingResult(null);
    setQprobeExact(null);
    setQprobeAdaptive(null);
  };

  const applyDemoPreset = (presetKey) => {
    const preset = DEMO_PRESETS[presetKey];
    setConfig({
      ...preset.config,
      qprobeTargets: [...GATE_SENSITIVE_TARGETS],
      qprobeTolerance: GATE_SENSITIVE_TOLERANCE,
    });
    setWorkflowResult(null);
    setRoutingResult(null);
    setQprobeExact(null);
    setQprobeAdaptive(null);
  };

  return (
    <div className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Crystal Forge</p>
          <h1>Quantum resource planner for correlated materials</h1>
          <p className="hero-text">
            Crystal Forge routes a hard materials query to the right compute stack, then compiles a smaller
            quantum execution workload for the observables that matter.
          </p>
          <div className="hero-metrics">
            {CORRMAP_RESULTS.map((item) => (
              <div key={item.label} className="hero-metric-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="hero-aside">
          <div className="hero-radar">
            <div className="hero-radar-grid" />
            <div className="hero-radar-ring hero-radar-ring-a" />
            <div className="hero-radar-ring hero-radar-ring-b" />
            <div className="hero-radar-ring hero-radar-ring-c" />
            <div className="hero-radar-dot hero-radar-dot-a" />
            <div className="hero-radar-dot hero-radar-dot-b" />
            <div className="hero-radar-dot hero-radar-dot-c" />
          </div>
          <div className="status-card">
            <p className="eyebrow">Hard-sector result</p>
            <strong>{workflowCoverage.safe.toFixed(1)}%</strong>
            <span>direct safe coverage at tolerance {tolerance.toFixed(2)}</span>
          </div>
          <div className="status-card secondary">
            <p className="eyebrow">Extended coverage</p>
            <strong>{workflowCoverage.recoverable.toFixed(1)}%</strong>
            <span>after structured decomposition and equivalence recovery</span>
          </div>
        </div>
      </section>

      {error ? (
        <section className="error-banner">
          <strong>Backend error:</strong> {error}
        </section>
      ) : null}

      <section className="workflow-sequence">
          <section className="panel">
              <div className="panel-head">
              <div>
                <p className="eyebrow">1. Launch setup</p>
                <h2>Choose a workload profile</h2>
              </div>
              <div className="button-cluster">
                <button onClick={applyParamsToState} disabled={pending}>
                  {pending ? "Running..." : "Apply params"}
                </button>
              </div>
            </div>
            <div className="button-cluster">
                <button type="button" onClick={() => applyDemoPreset("quantum")} disabled={pending}>
                  Quantum demo preset
                </button>
                <button type="button" onClick={() => applyDemoPreset("classical")} disabled={pending}>
                  Classical demo preset
                </button>
                <button type="button" onClick={() => applyDemoPreset("strongDrop")} disabled={pending}>
                  Strong gate drop
                </button>
                <button type="button" onClick={() => applyDemoPreset("noDrop")} disabled={pending}>
                  No gate drop
                </button>
              </div>
            <div className="slider-grid">
              {[
                ["t", 0.2, 2.0, 0.1],
                ["U", 1.0, 10.0, 0.5],
                ["mu", 0.0, 3.0, 0.1],
              ].map(([key, min, max, step]) => (
                <label key={key} className="slider-field">
                  <div className="slider-head">
                    <span>{key}</span>
                    <strong>{config.parameters[key].toFixed(1)}</strong>
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={config.parameters[key]}
                    onChange={(event) => updateParam(key, event.target.value)}
                  />
                </label>
              ))}
            </div>

            <div className="slider-grid compact">
              <label className="slider-field">
                <div className="slider-head">
                  <span>QProbe tolerance</span>
                  <strong>{tolerance.toFixed(2)}</strong>
                </div>
                <input
                  type="range"
                  min="0.03"
                  max="0.15"
                  step="0.01"
                  value={config.qprobeTolerance}
                  disabled
                  onChange={(event) => {
                    setConfig((current) => ({ ...current, qprobeTolerance: Number(event.target.value) }));
                    setQprobeExact(null);
                    setQprobeAdaptive(null);
                  }}
                />
              </label>
              <label className="slider-field">
                <div className="slider-head">
                  <span>Shots / group</span>
                  <strong>{config.qprobeShotsPerGroup}</strong>
                </div>
                <input
                  type="range"
                  min="1000"
                  max="4000"
                  step="500"
                  value={config.qprobeShotsPerGroup}
                  onChange={(event) => {
                    setConfig((current) => ({ ...current, qprobeShotsPerGroup: Number(event.target.value) }));
                    setQprobeExact(null);
                    setQprobeAdaptive(null);
                  }}
                />
              </label>
            </div>

            <div className="target-section">
              <p className="eyebrow">Live query bundle</p>
              <p className="workflow-note">
                This live demo is locked to <strong>D, K, Cs_max</strong> at tolerance <strong>0.15</strong> for
                computational feasibility during the demo.
              </p>
              <div className="target-grid">
                {GATE_SENSITIVE_TARGETS.map((key) => (
                  <div key={key} className="target-chip active curated">
                    <strong>{key}</strong>
                    <span>{TARGET_LABELS[key]}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

            <section className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">2. Lattice view</p>
                  <h2>Runtime state view</h2>
                </div>
              </div>
              {lattice ? (
                <>
                  <div className="lattice-grid-view" style={{ gridTemplateColumns: `repeat(${lattice.Lx}, minmax(0, 1fr))` }}>
                    {lattice.sites.map((site) => (
                      <div key={`${site.x}-${site.y}`} className="lattice-cell">
                        <span className="cell-coord">{site.x},{site.y}</span>
                        <strong>{(site.n_up + site.n_dn).toFixed(2)}</strong>
                        <small>sz {site.sz.toFixed(2)}</small>
                      </div>
                    ))}
                  </div>
                  <div className="phase-panel">
                    <div className="phase-panel-head">
                      <span className="phase-label">Current regime</span>
                      <strong>{phase?.label ?? "Unknown"}</strong>
                    </div>
                    <p className="phase-copy">
                      {phase?.label === "Antiferromagnet"
                        ? "Neighboring spins prefer to alternate, which signals strong competing magnetic order."
                        : phase?.label === "Mott Insulator"
                          ? "Interactions are dominating charge motion, making the state more localized and insulating."
                          : phase?.label === "Metal"
                            ? "Charge motion is relatively free, so the state looks more itinerant and less localized."
                            : phase?.label === "Singlet-rich"
                              ? "Short-range pairing tendencies are stronger, but this is not yet the same as proven superconductivity."
                              : "The backend has not produced a confident named regime yet."}
                    </p>
                    <div className="phase-stats">
                      <div className="phase-stat">
                        <span>Confidence</span>
                        <strong>{phase ? `${(phase.confidence * 100).toFixed(1)}%` : "n/a"}</strong>
                      </div>
                      <div className="phase-stat">
                        <span>Top signal</span>
                        <strong>{phase?.label ?? "n/a"}</strong>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <p className="workflow-note">Loading lattice...</p>
              )}
            </section>

            <section className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">3. Compute routing</p>
                  <h2>CorrMap</h2>
                </div>
                <button onClick={runCorrMap} disabled={pending}>
                  {pending ? "Running..." : "Run CorrMap"}
                </button>
              </div>
              <div className="metric-list">
                <MetricRow
                  label="Route"
                  value={corrmapRouteDisplay}
                />
                <MetricRow
                  label="Decision"
                  value={corrmapActionDisplay}
                />
                <MetricRow
                  label="Routing signal"
                  value={corrmapIntrinsicDisplay}
                />
                <MetricRow
                  label="Route confidence"
                  value={corrmapConfidenceDisplay}
                />
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">4. Measurement planning</p>
                  <h2>Execution workload compiler</h2>
                </div>
                <div className="button-cluster">
                  <button onClick={runExactQProbe} disabled={pending || !qprobeAllowed || !qprobeExactCompatible}>Run Exact</button>
                  <button onClick={runAdaptiveQProbe} disabled={pending || !qprobeAllowed || !qprobeAdaptiveCompatible}>Run Adaptive</button>
                </div>
              </div>
              <p className="workflow-note">
                {qprobeAllowed
                  ? "Quantum route active. QProbe compiles the smallest execution workload it can justify for the live query."
                  : "CorrMap kept this point in the classical scalable regime, so QProbe is disabled here."}
              </p>
              <p className="workflow-note">Format: `compiled / naive` execution workload.</p>
              {qprobeAllowed && (!qprobeExactCompatible || !qprobeAdaptiveCompatible) ? (
                <p className="workflow-note">
                  {`Current selection is not runnable in the live demo. Reduce to at most ${EXACT_DEMO_TARGET_LIMIT} of: ${LEGACY_QPROBE_TARGETS.join(", ")}.`}
                </p>
              ) : null}
              <div className="metric-list">
                <MetricRow
                  label="Naive workload"
                  value={qprobeExact ? `${qprobeExact.full_cost} circuit families` : qprobeAdaptive?.full_cost ? `${qprobeAdaptive.full_cost} circuit families` : qprobeAllowed ? "not run" : "disabled"}
                  hint={qprobeExactFullGateCost != null ? `${qprobeExactFullGateCost} basis-change gates` : qprobeAdaptive?.full_gate_cost != null ? `${qprobeAdaptive.full_gate_cost} basis-change gates` : undefined}
                />
                <MetricRow
                  label="Compiled exact workload"
                  value={
                    !qprobeAllowed
                      ? "disabled"
                      : !qprobeExactCompatible
                        ? "incompatible selection"
                    : qprobeExact
                      ? `${qprobeExact.recommended_cost} / ${qprobeExact.full_cost} circuit families`
                      : "not run"
                  }
                  hint={qprobeExactCompatible && qprobeExact ? `${qprobeExact.measurement_savings} circuit families saved` : undefined}
                />
                <MetricRow
                  label="Exact basis-change overhead"
                  value={
                    !qprobeAllowed
                      ? "disabled"
                      : !qprobeExactCompatible
                        ? "incompatible selection"
                        : qprobeExact && qprobeExactFullGateCost != null && qprobeExactRecommendedGateCost != null
                          ? `${qprobeExactRecommendedGateCost} / ${qprobeExactFullGateCost} gates`
                          : "not run"
                  }
                  hint={qprobeExactGateSavings != null ? `${qprobeExactGateSavings} rotation gates saved` : undefined}
                />
                <MetricRow
                  label="Compiled adaptive workload"
                  value={
                    !qprobeAllowed
                      ? "disabled"
                      : !qprobeAdaptiveCompatible
                        ? "incompatible selection"
                    : qprobeAdaptive?.predicted_cost
                      ? `${qprobeAdaptive.predicted_cost} / ${qprobeAdaptive.full_cost} circuit families`
                      : "not run"
                  }
                />
                <MetricRow
                  label="Adaptive basis-change overhead"
                  value={
                    !qprobeAllowed
                      ? "disabled"
                      : !qprobeAdaptiveCompatible
                        ? "incompatible selection"
                        : qprobeAdaptive?.predicted_gate_cost != null && qprobeAdaptive?.full_gate_cost != null
                          ? `${qprobeAdaptive.predicted_gate_cost} / ${qprobeAdaptive.full_gate_cost} gates`
                          : "not run"
                  }
                  hint={
                    qprobeAdaptive?.predicted_gate_cost != null && qprobeAdaptive?.full_gate_cost != null
                      ? `${qprobeAdaptive.full_gate_cost - qprobeAdaptive.predicted_gate_cost} rotation gates saved`
                      : undefined
                  }
                />
                <MetricRow
                  label="Adaptive viability"
                  value={
                    !qprobeAllowed
                      ? "disabled"
                      : !qprobeAdaptiveCompatible
                        ? "incompatible selection"
                    : qprobeAdaptive
                      ? (qprobeAdaptive.predicted_success ? "Likely within tolerance" : "Likely outside tolerance")
                      : "not run"
                  }
                />
                <MetricRow
                  label="Live query"
                  value={demoQprobeTargets.length > 0 ? demoQprobeTargets.join(", ") : "not demo-compatible"}
                />
              </div>
            </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">5. Execution diff</p>
                <h2>Execution impact</h2>
              </div>
            </div>
            {qprobeExact ? (
              <div className="impact-grid">
                <article className="impact-card primary">
                  <span className="impact-label">Circuit families</span>
                  <strong>{qprobeExact.recommended_cost} / {qprobeExact.full_cost}</strong>
                  <small>{qprobeExact.measurement_savings} families removed from the scheduled workload</small>
                </article>
                <article className="impact-card secondary">
                  <span className="impact-label">Basis-change gates</span>
                  <strong>{qprobeExactRecommendedGateCost} / {qprobeExactFullGateCost}</strong>
                  <small>{qprobeExactGateSavings ?? 0} gates removed before measurement</small>
                </article>
                <article className="impact-card accent">
                  <span className="impact-label">Execution reduction</span>
                  <strong>{qprobeCircuitReductionPct != null ? `${qprobeCircuitReductionPct.toFixed(0)}%` : "0%"}</strong>
                  <small>scheduled circuit-family reduction from naive execution</small>
                </article>
                <article className="impact-card accent">
                  <span className="impact-label">Gate overhead reduction</span>
                  <strong>{qprobeGateReductionPct != null ? `${qprobeGateReductionPct.toFixed(0)}%` : "0%"}</strong>
                  <small>basis-change gate reduction before readout</small>
                </article>
              </div>
            ) : (
              <p className="workflow-note">
                Run Exact to show how much execution workload QProbe removes from the naive plan.
              </p>
            )}
            {qprobeExact ? (
              <div className="impact-bars">
                <div className="impact-bar-row">
                  <div className="impact-bar-head">
                    <strong>Naive execution budget</strong>
                    <span>{qprobeExact.full_cost} families / {qprobeExactFullGateCost} gates</span>
                  </div>
                  <div className="impact-track">
                    <div className="impact-fill impact-fill-naive" style={{ width: "100%" }} />
                  </div>
                </div>
                <div className="impact-bar-row">
                  <div className="impact-bar-head">
                    <strong>Compiled exact budget</strong>
                    <span>{qprobeExact.recommended_cost} families / {qprobeExactRecommendedGateCost} gates</span>
                  </div>
                  <div className="impact-track">
                    <div
                      className="impact-fill impact-fill-compiled"
                      style={{ width: `${Math.max(qprobeCircuitReductionPct != null ? (100 - qprobeCircuitReductionPct) : 100, 6)}%` }}
                    />
                  </div>
                </div>
                {qprobeAdaptive?.predicted_cost != null && qprobeAdaptive?.full_gate_cost != null && qprobeAdaptive?.predicted_gate_cost != null ? (
                  <div className="impact-bar-row">
                    <div className="impact-bar-head">
                      <strong>Compiled adaptive budget</strong>
                      <span>{qprobeAdaptive.predicted_cost} families / {qprobeAdaptive.predicted_gate_cost} gates</span>
                    </div>
                    <div className="impact-track">
                      <div
                        className="impact-fill impact-fill-adaptive"
                        style={{ width: `${Math.max((qprobeAdaptive.predicted_cost / qprobeAdaptive.full_cost) * 100, 6)}%` }}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              null
            )}
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">6. Frontier coverage</p>
                <h2>Why the structured planner matters</h2>
              </div>
            </div>
            <div className="progress-stack">
              {BENCHMARK_SERIES.map((item) => (
                <ProgressBar
                  key={`${item.label}-${item.tolerance}`}
                  label={item.label}
                  meta={`tolerance ${item.tolerance}`}
                  tone={item.tone}
                  value={item.value}
                />
              ))}
            </div>
            <div className="recoverable-card">
              <strong>Extended coverage</strong>
              <p>
                `Safe` means the subproblem is directly handled by adaptive planning. `Extended`
                means the workflow can reuse equivalent successful subtargets to cover additional hard cases.
              </p>
              <div className="recoverable-grid">
                {RECOVERABLE_SERIES.map((item) => (
                  <div key={item.tolerance} className="recoverable-item">
                    <span>tol {item.tolerance}</span>
                    <strong>{item.value.toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </div>
          </section>

      </section>
    </div>
  );
}

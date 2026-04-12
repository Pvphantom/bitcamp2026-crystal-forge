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
  qprobeTargets: ["n", "D", "Ms2", "K", "Pair_span"],
  qprobeTolerance: 0.05,
  qprobeShotsPerGroup: 2000,
  qprobeReadoutFlipProb: 0.02,
  qprobeSeed: 17,
};

const MAX_QPROBE_TARGETS = 5;

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
  const [qprobeExact, setQprobeExact] = useState(null);
  const [qprobeAdaptive, setQprobeAdaptive] = useState(null);

  const tolerance = Number(config.qprobeTolerance);
  const selected = config.qprobeTargets;
  const activeChannels = useMemo(() => selectedChannels(selected), [selected]);
  const workflowCoverage = useMemo(() => workflowCoverageForTolerance(tolerance), [tolerance]);

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
      await loadState();
      setQprobeExact(null);
      setQprobeAdaptive(null);
      await runWorkflow(config);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  };

  const runCorrMap = async () => {
    await runWorkflow();
  };

  const runExactQProbe = async () => {
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/recommend-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: config.qprobeTargets,
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
    setPending(true);
    setError("");
    try {
      const result = await request("/api/qprobe/adaptive-plan", {
        method: "POST",
        body: JSON.stringify({
          targets: config.qprobeTargets,
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

  useEffect(() => {
    loadState();
    runWorkflow(DEFAULT_CONFIG);
  }, []);

  const routeSummary = workflowResult?.workflow_decision ?? null;
  const trustSummary = workflowResult?.trust ?? null;
  const routingSummary = workflowResult?.routing ?? null;
  const cheapObservables = workflowResult?.cheap_solver?.observables ?? {};
  const exactObservables = workflowResult?.exact_solver?.observables ?? {};
  const lattice = stateExport?.lattice ?? null;
  const phase = stateExport?.phase ?? null;

  const corrmapRouteLabel = routeSummary?.route_label ?? routingSummary?.route_label ?? null;
  const corrmapRouteDisplay =
    corrmapRouteLabel === "quantum_frontier"
      ? "Quantum frontier"
      : corrmapRouteLabel === "scalable_classical"
        ? "Classical scalable"
        : corrmapRouteLabel === "mean_field"
          ? "Legacy mean field"
          : "Waiting for backend";
  const corrmapActionDisplay =
    corrmapRouteLabel === "quantum_frontier"
      ? "Escalate to quantum workflow and QProbe"
      : corrmapRouteLabel === "scalable_classical"
        ? "Stay in scalable classical simulation"
        : routeSummary?.recommendation ?? routingSummary?.recommended_action ?? "n/a";
  const corrmapReasonDisplay =
    routingSummary?.intrinsic_label ??
    (routingSummary?.abstained ? routingSummary.abstain_reason : null) ??
    "binary Hubbard routing";

  const updateParam = (key, value) => {
    setConfig((current) => ({
      ...current,
      parameters: {
        ...current.parameters,
        [key]: Number(value),
      },
    }));
  };

  const toggleTarget = (target) => {
    setConfig((current) => {
      const has = current.qprobeTargets.includes(target);
      const nextTargets = has
        ? current.qprobeTargets.filter((entry) => entry !== target)
        : current.qprobeTargets.length >= MAX_QPROBE_TARGETS
          ? [...current.qprobeTargets.slice(1), target]
          : [...current.qprobeTargets, target];
      return {
        ...current,
        qprobeTargets: nextTargets.length > 0 ? nextTargets : current.qprobeTargets,
      };
    });
  };

  return (
    <div className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Crystal Forge</p>
          <h1>Hubbard superconductivity workflow</h1>
          <p className="hero-text">
            This demo now focuses on one story only: use CorrMap to route a superconductivity-relevant
            Hubbard instance, then show how the measurement workflow decomposes hard pairing and transport
            observables into safer subproblems.
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
            <p className="eyebrow">Recoverable coverage</p>
            <strong>{workflowCoverage.recoverable.toFixed(1)}%</strong>
            <span>after symmetry / orbit closure in the structured workflow</span>
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
                <h2>Choose a Hubbard superconductivity panel</h2>
              </div>
              <div className="button-cluster">
                <button onClick={applyParamsToState} disabled={pending}>
                  {pending ? "Running..." : "Apply params"}
                </button>
                <button onClick={() => runWorkflow()} disabled={pending}>
                  Refresh workflow
                </button>
              </div>
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
                  max="0.10"
                  step="0.01"
                  value={config.qprobeTolerance}
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, qprobeTolerance: Number(event.target.value) }))
                  }
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
                  onChange={(event) =>
                    setConfig((current) => ({ ...current, qprobeShotsPerGroup: Number(event.target.value) }))
                  }
                />
              </label>
            </div>

            <div className="target-section">
              <p className="eyebrow">Requested observables</p>
              <p className="workflow-note">
                The live QProbe backend currently accepts at most {MAX_QPROBE_TARGETS} targets per request, so the
                selector keeps the workflow inside that production budget.
              </p>
              <div className="target-grid">
                {Object.entries(TARGET_LABELS).map(([key, label]) => {
                  const active = selected.includes(key);
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`target-chip ${active ? "active" : ""}`}
                      onClick={() => toggleTarget(key)}
                    >
                      <strong>{key}</strong>
                      <span>{label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

            <section className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">2. Lattice view</p>
                  <h2>Actual 2x2 state</h2>
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
                  <p className="eyebrow">3. Route decision</p>
                  <h2>CorrMap</h2>
                </div>
                <button onClick={runCorrMap} disabled={pending}>Run CorrMap</button>
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
                  label="Intrinsic read"
                  value={corrmapReasonDisplay}
                />
                <MetricRow
                  label="Cheap solver error"
                  value={trustSummary ? formatNumber(trustSummary.max_abs_error) : "n/a"}
                  hint={trustSummary?.risk_label ? `legacy trust: ${trustSummary.risk_label}` : undefined}
                />
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">4. Measurement planning</p>
                  <h2>QProbe</h2>
                </div>
                <div className="button-cluster">
                  <button onClick={runExactQProbe} disabled={pending}>Run Exact</button>
                  <button onClick={runAdaptiveQProbe} disabled={pending}>Run Adaptive</button>
                </div>
              </div>
              <div className="metric-list">
                <MetricRow
                  label="Exact savings"
                  value={qprobeExact ? `${qprobeExact.measurement_savings} groups` : "not run"}
                />
                <MetricRow
                  label="Adaptive savings"
                  value={qprobeAdaptive ? `${qprobeAdaptive.measurement_savings} groups` : "not run"}
                />
                <MetricRow
                  label="Adaptive oracle check"
                  value={qprobeAdaptive ? (qprobeAdaptive.oracle_benchmark_within_tolerance ? "Within tolerance" : "Outside tolerance") : "not run"}
                />
                <MetricRow
                  label="Targets"
                  value={`${config.qprobeTargets.length} / ${MAX_QPROBE_TARGETS}`}
                />
              </div>
            </section>

          <MissionMap routeSummary={routeSummary} activeChannels={activeChannels} />

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">5. Signal snapshot</p>
                <h2>Cheap solver vs exact reference</h2>
              </div>
            </div>
            <div className="observables-grid">
              {Object.keys(cheapObservables).slice(0, 7).map((name) => (
                <div key={name} className="observable-card">
                  <span>{TARGET_LABELS[name] ?? name}</span>
                  <strong>{formatNumber(cheapObservables[name])}</strong>
                  <small>exact {formatNumber(exactObservables[name])}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">6. Signal constellation</p>
                <h2>Channel-first measurement plan</h2>
              </div>
            </div>
            <div className="channel-grid">
              {activeChannels.map((channel) => {
                const info = benchmarkBackedChannelStatus(channel, tolerance);
                return (
                  <article key={channel} className={`channel-card tone-${info.tone}`}>
                    <div className="channel-top">
                      <div className="channel-title">
                        <span className={`channel-glyph glyph-${channel}`}>
                          <ChannelIcon channel={channel} />
                        </span>
                        <strong>{channel[0].toUpperCase() + channel.slice(1)}</strong>
                      </div>
                      <span className={`channel-pill pill-${info.tone}`}>{info.status}</span>
                    </div>
                    <div className="channel-targets">
                      {TARGET_GROUPS[channel]
                        .filter((name) => selected.includes(name))
                        .map((name) => (
                          <span key={name}>{name}</span>
                        ))}
                    </div>
                    <p>{info.note}</p>
                  </article>
                );
              })}
            </div>
            <div className="workflow-note">
              The workflow keeps charge and spin as direct channels, then decomposes hard transport and
              pairing observables into physically coherent transverse/Z subtargets before adaptive planning.
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">7. Frontier coverage</p>
                <h2>Why the structured workflow matters</h2>
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
              <strong>Recoverable coverage</strong>
              <p>
                ``Safe'' means the subproblem is directly handled by adaptive planning. ``Recoverable''
                means the workflow can still use symmetry-equivalent successful subtargets to stand in for
                unresolved ones.
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

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">8. Mission framing</p>
                <h2>How to present this result</h2>
              </div>
            </div>
            <ul className="story-list">
              <li>CorrMap works as the classical-vs-quantum routing layer for Hubbard superconductivity studies.</li>
              <li>Exact QProbe remains the tractable-regime oracle.</li>
              <li>Generic adaptive planning fails on hard mixed superconductivity bundles.</li>
              <li>The superconductivity workflow makes real progress by channelizing and decomposing the hard sector.</li>
              <li>At moderate tolerance, the hard sector is close to mostly-working rather than completely failing.</li>
            </ul>
          </section>
      </section>
    </div>
  );
}

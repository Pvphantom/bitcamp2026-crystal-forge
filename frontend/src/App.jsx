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
  Mz: "Average Z magnetization",
  Mx: "Average X magnetization",
  ZZ_nn: "Nearest-neighbor ZZ order",
  Mstag2: "Staggered Z order strength",
  Z_span: "Long-range Z link",
};

const TARGET_EXPLANATIONS = {
  D: "How often two electrons sit on the same site.",
  n: "How many electrons are present on average per site.",
  Ms2: "How strongly spins line up in an alternating antiferromagnetic pattern.",
  K: "How strongly electrons are delocalizing and hopping across the lattice.",
  Cs_max: "How correlated the most distant sites are.",
  Mz: "Average spin polarization along the Z direction.",
  Mx: "Average spin polarization along the transverse X direction.",
  ZZ_nn: "How strongly neighboring spins align in the Ising interaction direction.",
  Mstag2: "How strongly checkerboard-like Z order appears.",
  Z_span: "How correlated the most distant spins are along Z.",
};

const MODEL_FAMILY_LABELS = {
  hubbard: "Fermi-Hubbard",
  tfim: "Transverse-field Ising",
};

const WORKFLOW_TARGET_OPTIONS = {
  hubbard: ["D", "n", "Ms2", "K", "Cs_max"],
  tfim: ["Mz", "Mx", "ZZ_nn", "Mstag2", "Z_span"],
};

const WORKFLOW_PRESETS = {
  tfim_safe: {
    label: "TFIM Safe",
    description: "The cheap solver is accurate enough, so the workflow should stop before the quantum path.",
    config: {
      modelFamily: "tfim",
      Lx: 2,
      Ly: 2,
      parameters: { J: 0.1, h: 0.5, g: 1.0 },
      qprobeTargets: ["Mz", "Mx", "ZZ_nn"],
      qprobeTolerance: 0.03,
      qprobeShotsPerGroup: 4000,
      qprobeReadoutFlipProb: 0.02,
      qprobeSeed: 7,
    },
  },
  tfim_quantum: {
    label: "TFIM Quantum Escalation",
    description: "The cheap solver fails, so the workflow escalates to VQE and then applies QProbe to the quantum path.",
    config: {
      modelFamily: "tfim",
      Lx: 2,
      Ly: 2,
      parameters: { J: 1.0, h: 0.8, g: 0.0 },
      qprobeTargets: ["Mz", "ZZ_nn", "Mstag2"],
      qprobeTolerance: 0.03,
      qprobeShotsPerGroup: 4000,
      qprobeReadoutFlipProb: 0.02,
      qprobeSeed: 7,
    },
  },
  hubbard_fallback: {
    label: "Hubbard Exact Fallback",
    description: "The cheap solver looks risky, but no quantum solver is registered yet, so the workflow falls back to the exact oracle.",
    config: {
      modelFamily: "hubbard",
      Lx: 2,
      Ly: 2,
      parameters: { t: 1.0, U: 4.0, mu: 2.0 },
      qprobeTargets: ["D", "Ms2", "Cs_max"],
      qprobeTolerance: 0.03,
      qprobeShotsPerGroup: 2000,
      qprobeReadoutFlipProb: 0.01,
      qprobeSeed: 7,
    },
  },
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

const BACKEND_TEST_GUIDE = [
  {
    title: "State backend",
    endpoint: "GET /api/state/export",
    body: "Loads the current lattice, observables, phase summary, and saved model metrics. Use Refresh to verify the backend is alive and the frontend is synchronized.",
  },
  {
    title: "Trust backend",
    endpoint: "POST /api/trust/evaluate",
    body: "Runs CorrMap on the current state. This compares the cheap solver and exact solver, reports observable gaps, and returns the trust recommendation.",
  },
  {
    title: "Measurement backend",
    endpoint: "POST /api/qprobe/recommend-plan + POST /api/qprobe/adaptive-plan",
    body: "Runs the exact fixed-plan search and the scalable adaptive runtime policy. Use these to test measurement compression, safety refusals, and adaptive stopping.",
  },
];

const PARAMETER_HELP = [
  {
    key: "t",
    title: "t: hopping strength",
    body: "Increase this when you want electrons to move more freely between neighboring sites. Larger t usually makes motion-related signals stronger and can make the state look more metal-like.",
  },
  {
    key: "U",
    title: "U: on-site repulsion",
    body: "Increase this when you want to punish two electrons for sharing the same site. Larger U usually lowers double occupancy and can make simple cheap solvers less trustworthy.",
  },
  {
    key: "mu",
    title: "μ: filling control",
    body: "Move μ to change how full the lattice wants to be. Around half filling, the average occupancy n is near 1. Moving away from that often changes both the regime summary and the solver-trust decision.",
  },
];

const FRONTEND_TEST_CHECKLIST = [
  "Refresh the page and confirm the state, trust panel, and library all load without errors.",
  "Change one slider, click Apply Params, and verify the observables and trust panel update together.",
  "Press Exact Ground State to get the cleanest reference state before testing CorrMap or QProbe.",
  "Run Compression Win to verify the fixed QProbe planner finds a cheaper safe plan.",
  "Run Adaptive Win to verify the adaptive planner stops from uncertainty/coverage, not from oracle knowledge.",
  "Run No Safe Shortcut to verify the planner refuses unsafe compression rather than forcing a win.",
];

const PANEL_EXPLANATIONS = {
  regime:
    "This section is the fast interpretation layer. It tells you what kind of state the backend thinks you created, but it is not the final product decision by itself.",
  corrmap:
    "This section is the trust layer. It checks whether the cheap approximation is good enough or whether strong correlations make that shortcut unreliable.",
  qprobe:
    "This section is the action layer. Once you know which solver path you trust, it tells you how much measurement cost you can safely remove.",
};

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

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function formatSolverName(name) {
  const labels = {
    exact_ed: "Exact ED",
    mean_field: "Mean-field",
    tfim_mean_field: "TFIM Mean-field",
    vqe: "Variational Quantum Eigensolver",
  };
  return labels[name] ?? name ?? "Unavailable";
}

function modelParameterKeys(modelFamily) {
  return modelFamily === "tfim" ? ["J", "h", "g"] : ["t", "U", "mu"];
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
  const [workflowPending, setWorkflowPending] = useState(false);
  const [workflowError, setWorkflowError] = useState("");
  const [workflowPresetKey, setWorkflowPresetKey] = useState("tfim_quantum");
  const [workflowResult, setWorkflowResult] = useState(null);
  const [workflowConfig, setWorkflowConfig] = useState(WORKFLOW_PRESETS.tfim_quantum.config);
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
    runWorkflowAnalysis(WORKFLOW_PRESETS.tfim_quantum.config, "tfim_quantum");
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

  const runWorkflowAnalysis = async (config = workflowConfig, presetKey = workflowPresetKey) => {
    setWorkflowPending(true);
    setWorkflowError("");
    try {
      const result = await request("/api/workflow/analyze", {
        method: "POST",
        body: JSON.stringify({
          model_family: config.modelFamily,
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
      setWorkflowResult(result);
      setWorkflowPresetKey(presetKey ?? null);
    } catch (err) {
      setWorkflowError(err.message);
    } finally {
      setWorkflowPending(false);
    }
  };

  const loadWorkflowPreset = async (presetKey) => {
    const nextConfig = { ...WORKFLOW_PRESETS[presetKey].config, parameters: { ...WORKFLOW_PRESETS[presetKey].config.parameters }, qprobeTargets: [...WORKFLOW_PRESETS[presetKey].config.qprobeTargets] };
    setWorkflowConfig(nextConfig);
    await runWorkflowAnalysis(nextConfig, presetKey);
  };

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
  const workflowPreset = workflowPresetKey ? WORKFLOW_PRESETS[workflowPresetKey] : null;
  const workflowTargets = WORKFLOW_TARGET_OPTIONS[workflowConfig.modelFamily];

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

      <section className="inspector-grid">
        {BACKEND_TEST_GUIDE.map((entry) => (
          <article key={entry.title} className="inspector-card">
            <span className="eyebrow">Backend Path</span>
            <strong>{entry.title}</strong>
            <code>{entry.endpoint}</code>
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

      <section className="workflow-card">
        <div>
          <p className="eyebrow">How To Test Everything</p>
          <h2>Use the page as a backend inspection console</h2>
        </div>
        <div className="checklist-grid">
          {FRONTEND_TEST_CHECKLIST.map((item) => (
            <div key={item} className="workflow-item">
              <span>Check</span>
              <strong>{item}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="workflow-tester panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Unified Backend Route</p>
            <h2>Test the full final backend directly</h2>
          </div>
          <span className="status-pill">{workflowPending ? "Running" : "Ready"}</span>
        </div>
        <p className="panel-note panel-note-tight">
          This section hits <code>POST /api/workflow/analyze</code>, which is the final backend
          route tying together cheap solvers, exact oracles, CorrMap trust routing, TFIM VQE,
          and QProbe only when the workflow actually escalates to the quantum path.
        </p>

        <div className="preset-strip">
          {Object.entries(WORKFLOW_PRESETS).map(([key, preset]) => (
            <button key={key} className="preset-card" onClick={() => loadWorkflowPreset(key)} disabled={workflowPending}>
              <span className="eyebrow">Workflow Preset</span>
              <strong>{preset.label}</strong>
              <span>{preset.description}</span>
            </button>
          ))}
        </div>

        {workflowPreset ? (
          <section className="preset-note">
            <strong>{workflowPreset.label}</strong>
            <span>{workflowPreset.description}</span>
          </section>
        ) : null}

        <div className="dashboard-grid">
          <article className="panel controls-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Workflow Inputs</p>
                <h2>Model, solver, and measurement inputs</h2>
              </div>
            </div>
            <div className="endpoint-box">
              <strong>Backend route exercised here</strong>
              <code>POST /api/workflow/analyze</code>
              <span>
                This is the route you should use to test the real final backend. It replaces the old
                pattern of mentally stitching together separate endpoints for trust, VQE, and QProbe.
              </span>
            </div>

            <div className="board-controls">
              <label>
                Model family
                <select
                  value={workflowConfig.modelFamily}
                  onChange={(event) => {
                    const modelFamily = event.target.value;
                    setWorkflowPresetKey(null);
                    setWorkflowConfig((current) => ({
                      ...current,
                      modelFamily,
                      parameters: modelFamily === "tfim" ? { J: 1.0, h: 0.8, g: 0.0 } : { t: 1.0, U: 4.0, mu: 2.0 },
                      qprobeTargets: [...WORKFLOW_TARGET_OPTIONS[modelFamily].slice(0, 3)],
                    }));
                  }}
                  disabled={workflowPending}
                >
                  <option value="hubbard">Fermi-Hubbard</option>
                  <option value="tfim">Transverse-field Ising</option>
                </select>
              </label>
              <button onClick={() => runWorkflowAnalysis()} disabled={workflowPending}>Run Full Workflow</button>
            </div>

            <div className="target-help-list">
              {modelParameterKeys(workflowConfig.modelFamily).map((key) => (
                <div key={key} className="target-help-row">
                  <strong>{key}</strong>
                  <span>
                    {workflowConfig.modelFamily === "tfim"
                      ? key === "J"
                        ? "Nearest-neighbor Ising interaction strength."
                        : key === "h"
                          ? "Transverse X-field strength. This is what competes against ZZ ordering."
                          : "Longitudinal Z-field strength that biases the spins."
                      : key === "t"
                        ? "Electron hopping strength."
                        : key === "U"
                          ? "On-site repulsion strength."
                          : "Chemical potential controlling lattice filling."}
                  </span>
                </div>
              ))}
            </div>

            <div className="slider-stack">
              {modelParameterKeys(workflowConfig.modelFamily).map((key) => {
                const range =
                  workflowConfig.modelFamily === "tfim"
                    ? { J: { min: 0, max: 2, step: 0.1 }, h: { min: 0, max: 3, step: 0.1 }, g: { min: 0, max: 2, step: 0.1 } }[key]
                    : { t: { min: 0.5, max: 2, step: 0.1 }, U: { min: 0, max: 10, step: 0.1 }, mu: { min: -2, max: 5, step: 0.1 } }[key];
                return (
                  <label key={key}>
                    <span>{key}</span>
                    <input
                      type="range"
                      min={range.min}
                      max={range.max}
                      step={range.step}
                      value={workflowConfig.parameters[key]}
                      onChange={(event) =>
                        setWorkflowConfig((current) => ({
                          ...current,
                          parameters: {
                            ...current.parameters,
                            [key]: Number(event.target.value),
                          },
                        }))
                      }
                      disabled={workflowPending}
                    />
                    <strong>{Number(workflowConfig.parameters[key]).toFixed(1)}</strong>
                  </label>
                );
              })}
            </div>

            <div className="target-grid">
              {workflowTargets.map((target) => (
                <button
                  key={target}
                  className={`target-chip ${workflowConfig.qprobeTargets.includes(target) ? "target-active" : ""}`}
                  onClick={() =>
                    setWorkflowConfig((current) => {
                      const present = current.qprobeTargets.includes(target);
                      const nextTargets = present
                        ? current.qprobeTargets.filter((entry) => entry !== target)
                        : [...current.qprobeTargets, target];
                      return {
                        ...current,
                        qprobeTargets: nextTargets.length > 0 ? nextTargets : current.qprobeTargets,
                      };
                    })
                  }
                  disabled={workflowPending}
                >
                  {TARGET_LABELS[target]}
                </button>
              ))}
            </div>

            <div className="slider-stack">
              <label>
                <span>Workflow QProbe tolerance</span>
                <input
                  type="range"
                  min="0.005"
                  max="0.1"
                  step="0.005"
                  value={workflowConfig.qprobeTolerance}
                  onChange={(event) => setWorkflowConfig((current) => ({ ...current, qprobeTolerance: Number(event.target.value) }))}
                  disabled={workflowPending}
                />
                <strong>{workflowConfig.qprobeTolerance.toFixed(3)}</strong>
              </label>
              <label>
                <span>Workflow shots per group</span>
                <input
                  type="range"
                  min="500"
                  max="20000"
                  step="500"
                  value={workflowConfig.qprobeShotsPerGroup}
                  onChange={(event) => setWorkflowConfig((current) => ({ ...current, qprobeShotsPerGroup: Number(event.target.value) }))}
                  disabled={workflowPending}
                />
                <strong>{workflowConfig.qprobeShotsPerGroup}</strong>
              </label>
              <label>
                <span>Workflow readout noise</span>
                <input
                  type="range"
                  min="0"
                  max="0.15"
                  step="0.01"
                  value={workflowConfig.qprobeReadoutFlipProb}
                  onChange={(event) => setWorkflowConfig((current) => ({ ...current, qprobeReadoutFlipProb: Number(event.target.value) }))}
                  disabled={workflowPending}
                />
                <strong>{workflowConfig.qprobeReadoutFlipProb.toFixed(2)}</strong>
              </label>
            </div>
          </article>

          <article className="panel metrics-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Workflow Output</p>
                <h2>What the final backend decided</h2>
              </div>
            </div>
            {workflowError ? (
              <section className="error-banner">
                <strong>Workflow error:</strong> {workflowError}
              </section>
            ) : null}

            {workflowResult ? (
              <>
                <div className={`qprobe-banner ${workflowResult.workflow_decision.escalation_triggered ? "qprobe-failure" : "qprobe-success"}`}>
                  <strong>{workflowResult.workflow_decision.escalation_triggered ? "Escalation triggered" : "Cheap solver accepted"}</strong>
                  <span>{workflowResult.workflow_decision.recommendation}</span>
                </div>

                <div className="metric-grid">
                  <div className="metric-card">
                    <span>Model family</span>
                    <strong>{MODEL_FAMILY_LABELS[workflowResult.model_family] ?? workflowResult.model_family}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Cheap solver</span>
                    <strong>{formatSolverName(workflowResult.selected_cheap_solver)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Strong solver</span>
                    <strong>{formatSolverName(workflowResult.selected_strong_solver)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Active solver</span>
                    <strong>{formatSolverName(workflowResult.workflow_decision.active_solver)}</strong>
                  </div>
                  <div className="metric-card">
                    <span>Measurement mode</span>
                    <strong>{workflowResult.workflow_decision.measurement_mode}</strong>
                  </div>
                  <div className="metric-card">
                    <span>CorrMap risk</span>
                    <strong>{workflowResult.trust.risk_label}</strong>
                  </div>
                </div>

                <details className="details-box" open>
                  <summary>Exactly how to read the unified workflow result</summary>
                  <div className="details-content">
                    <p>
                      <strong>Cheap solver</strong> is the fast classical approximation. <strong>Exact oracle</strong>
                      is only the benchmark reference on these tiny solvable systems. If CorrMap says the cheap solver
                      is safe, the workflow stops there. If CorrMap says the cheap solver is risky and a strong quantum
                      solver exists, the workflow escalates into VQE.
                    </p>
                    <p>
                      Only after that escalation does QProbe run. That is the key product logic tying everything together:
                      CorrMap decides <em>whether</em> to go quantum, and QProbe decides <em>how</em> to measure the
                      quantum result efficiently.
                    </p>
                  </div>
                </details>

                <div className="comparison-grid">
                  <div className="comparison-card">
                    <p className="eyebrow">Cheap Solver</p>
                    <h3>{formatSolverName(workflowResult.cheap_solver.solver_name)}</h3>
                    <div className="qprobe-table">
                      {Object.entries(workflowResult.cheap_solver.observables).map(([name, value]) => (
                        <div key={name} className="qprobe-row">
                          <span>{TARGET_LABELS[name] ?? name}</span>
                          <strong>{Number(value).toFixed(4)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="comparison-card">
                    <p className="eyebrow">Exact Oracle</p>
                    <h3>{formatSolverName(workflowResult.exact_solver.solver_name)}</h3>
                    <div className="qprobe-table">
                      {Object.entries(workflowResult.exact_solver.observables).map(([name, value]) => (
                        <div key={name} className="qprobe-row">
                          <span>{TARGET_LABELS[name] ?? name}</span>
                          <strong>{Number(value).toFixed(4)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {workflowResult.strong_solver ? (
                  <div className="comparison-card">
                    <p className="eyebrow">Strong Solver</p>
                    <h3>{formatSolverName(workflowResult.strong_solver.solver_name)}</h3>
                    <p className="panel-note panel-note-tight">
                      This solver is only activated when the workflow escalates beyond the cheap path.
                    </p>
                    <div className="qprobe-table">
                      {Object.entries(workflowResult.strong_solver.observables).map(([name, value]) => (
                        <div key={name} className="qprobe-row">
                          <span>{TARGET_LABELS[name] ?? name}</span>
                          <strong>{Number(value).toFixed(4)}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                <div className="qprobe-table">
                  {Object.keys(workflowResult.trust.abs_error).map((name) => (
                    <div key={name} className="qprobe-row">
                      <span>{TARGET_LABELS[name] ?? name}</span>
                      <span>abs gap {workflowResult.trust.abs_error[name].toFixed(4)}</span>
                      <strong>rel gap {workflowResult.trust.rel_error[name].toFixed(3)}</strong>
                    </div>
                  ))}
                </div>

                {workflowResult.qprobe_exact ? (
                  <>
                    <div className={`qprobe-banner ${workflowResult.qprobe_exact.success ? "qprobe-success" : "qprobe-failure"}`}>
                      <strong>QProbe is active on the quantum path</strong>
                      <span>
                        Planning state: {formatSolverName(workflowResult.qprobe_exact.planning_state_solver)} ·
                        oracle reference: {formatSolverName(workflowResult.qprobe_exact.oracle_reference_solver)}
                      </span>
                    </div>

                    <div className="metric-grid">
                      <div className="metric-card">
                        <span>Full plan cost</span>
                        <strong>{workflowResult.qprobe_exact.full_cost}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Recommended plan cost</span>
                        <strong>{workflowResult.qprobe_exact.recommended_cost}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Saved groups</span>
                        <strong>{workflowResult.qprobe_exact.measurement_savings}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Worst error</span>
                        <strong>{workflowResult.qprobe_exact.max_abs_error.toFixed(4)}</strong>
                      </div>
                    </div>

                    <div className="comparison-grid">
                      <div className="comparison-card">
                        <p className="eyebrow">Fixed QProbe</p>
                        <h3>{workflowResult.qprobe_exact.message}</h3>
                        <div className="group-list">
                          {workflowResult.qprobe_exact.recommended_groups.map((group) => (
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
                        <p className="eyebrow">Adaptive QProbe</p>
                        <h3>{workflowResult.qprobe_adaptive.message}</h3>
                        <div className="qprobe-table">
                          <div className="qprobe-row">
                            <span>Runtime stop rule</span>
                            <strong>{workflowResult.qprobe_adaptive.runtime_stop_rule}</strong>
                          </div>
                          <div className="qprobe-row">
                            <span>Final cost</span>
                            <strong>{workflowResult.qprobe_adaptive.final_cost}</strong>
                          </div>
                          <div className="qprobe-row">
                            <span>Oracle benchmark</span>
                            <strong>{workflowResult.qprobe_adaptive.oracle_benchmark_within_tolerance ? "Within tolerance" : "Outside tolerance"}</strong>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="scope-box">
                    <strong>QProbe intentionally inactive</strong>
                    <span>
                      The workflow did not escalate into the quantum path here, so QProbe was not run. That is the
                      correct final backend behavior.
                    </span>
                  </div>
                )}

                <details className="details-box">
                  <summary>Raw unified backend JSON</summary>
                  <div className="details-content">
                    <pre className="json-box">{prettyJson(workflowResult)}</pre>
                  </div>
                </details>
              </>
            ) : (
              <p className="panel-note">Run the unified workflow to inspect the final backend route.</p>
            )}
          </article>
        </div>
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
          <div className="endpoint-box">
            <strong>Backend routes this panel exercises</strong>
            <code>POST /api/state/place-configuration</code>
            <code>POST /api/state/reset-neel</code>
            <code>GET /api/state/export</code>
            <span>
              Use this panel to test state construction. After you apply the board,
              the site occupations below should change immediately, and the regime,
              trust, and observable panels should all reflect the new state.
            </span>
          </div>

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
            <div className="help-row">
              <strong>What to verify here</strong>
              <span>
                If you change a tile from <code>empty</code> to <code>double</code>,
                the displayed <code>n↑</code>, <code>n↓</code>, and <code>D</code> values
                for that site should update after Apply Board.
              </span>
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
          <div className="endpoint-box">
            <strong>Backend routes this panel exercises</strong>
            <code>POST /api/state/set-params</code>
            <code>POST /api/state/evolve</code>
            <code>POST /api/state/create</code>
            <code>POST /api/state/ground-state</code>
            <span>
              Use this panel to test Hamiltonian changes. The safest backend test is:
              change one slider, click Apply Params, then click Exact Ground State and
              watch the observables, regime summary, and CorrMap output all change together.
            </span>
          </div>

          <div className="target-help-list">
            {PARAMETER_HELP.map((entry) => (
              <div key={entry.key} className="target-help-row">
                <strong>{entry.title}</strong>
                <span>{entry.body}</span>
              </div>
            ))}
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
            <div className="help-row">
              <strong>Recommended backend test</strong>
              <span>
                Start from Exact Ground State, then change only <code>U</code>. Larger
                <code>U</code> should usually lower <code>D</code> and make CorrMap more
                suspicious of the cheap solver.
              </span>
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
          <div className="endpoint-box">
            <strong>Backend routes this panel exercises</strong>
            <code>GET /api/state/export</code>
            <code>POST /api/state/predict-phase</code>
            <span>
              This panel is the interpretation layer. It tells you what state you are
              looking at, but it is not the trust or action decision. Use it to answer:
              “What regime am I in?” before reading CorrMap or QProbe.
            </span>
          </div>
          <p className="panel-note panel-note-tight">{PANEL_EXPLANATIONS.regime}</p>

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
          <details className="details-box">
            <summary>Exactly how to read this panel</summary>
            <div className="details-content">
              <p>
                The label is a summary, not a proof. The probability bars show how strongly
                the backend classifier prefers one regime over the others. If the confidence
                is low or several bars are close together, treat this as a soft interpretation.
              </p>
              <p>
                Use this panel first for orientation, then move to CorrMap to decide whether
                the cheap approximation is trustworthy, and then to QProbe to decide how to
                measure the stronger path.
              </p>
            </div>
          </details>
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
          <div className="endpoint-box">
            <strong>Backend routes this panel exercises</strong>
            <code>POST /api/trust/evaluate</code>
            <code>GET /api/trust/metrics</code>
            <span>
              CorrMap is the solver-routing layer. It compares the cheap mean-field
              answer against the exact ED answer on small solvable systems, then uses
              that training signal to predict whether the cheap approximation is safe.
            </span>
          </div>
          <p className="panel-note panel-note-tight">{PANEL_EXPLANATIONS.corrmap}</p>
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

              <details className="details-box">
                <summary>Exactly how to read CorrMap</summary>
                <div className="details-content">
                  <p>
                    <strong>Cheap solver</strong> is the fast approximation. <strong>Exact solver</strong>
                    is the reference answer on this tiny solvable lattice. The observable gaps below
                    show how wrong the cheap answer is for each physics signal.
                  </p>
                  <p>
                    If the largest gap is small, CorrMap can call the cheap solver <em>safe</em>.
                    If the gaps are moderate, it returns <em>warning</em>. If the gaps are large,
                    it returns <em>unsafe</em> and recommends escalation.
                  </p>
                  <p>
                    This is why CorrMap matters: it answers “Can I trust the fast approximation,
                    or do I need the stronger solver path?”
                  </p>
                </div>
              </details>

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
          <div className="endpoint-box">
            <strong>Backend routes this panel exercises</strong>
            <code>GET /api/qprobe/library</code>
            <code>POST /api/qprobe/recommend-plan</code>
            <code>POST /api/qprobe/adaptive-plan</code>
            <span>
              This is the measurement-planning layer. The fixed planner finds the best
              safe static plan. The adaptive planner makes runtime decisions using only
              target coverage and uncertainty, then the exact oracle validates that
              decision afterward on the small testbed.
            </span>
          </div>
          <p className="panel-note panel-note-tight">{PANEL_EXPLANATIONS.qprobe}</p>
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
            <div className="help-row">
              <strong>How to interpret the sliders</strong>
              <span>
                Smaller tolerance makes QProbe stricter. More measurements per group
                lowers shot noise. Higher noise level simulates a worse device and usually
                makes compression harder.
              </span>
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
              <details className="details-box">
                <summary>Exactly how to read fixed QProbe</summary>
                <div className="details-content">
                  <p>
                    <strong>Full Plan</strong> is the “measure everything needed” baseline.
                    <strong> QProbe Plan</strong> is the smaller plan the oracle search found.
                  </p>
                  <p>
                    If <strong>Groups saved</strong> is positive and <strong>Worst error</strong>
                    is still below tolerance, then the shortcut was successful. If QProbe refuses
                    to compress, that is not a bug. It means the current noise budget or target
                    list makes the shortcut unsafe.
                  </p>
                  <p>
                    ML-QProbe is the learned approximation to the exact planner. Use the cost and
                    safety match pills to see whether the learned model agrees with the exact planner.
                  </p>
                </div>
              </details>
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
              <details className="details-box">
                <summary>Exactly how to read Adaptive QProbe</summary>
                <div className="details-content">
                  <p>
                    Adaptive QProbe is the scalable version. At runtime it does <strong>not</strong>
                    look at the exact oracle. It only looks at which targets are already covered and
                    how uncertain the reconstructed signals still are.
                  </p>
                  <p>
                    The <strong>Oracle benchmark</strong> fields are shown only because this small
                    system is exactly solvable. They let you verify after the fact whether the runtime
                    stop decision was actually safe.
                  </p>
                  <p>
                    Each step below tells you which measurement group was chosen, which targets it
                    supports, what remained unresolved, and how the runtime uncertainty changed.
                  </p>
                </div>
              </details>
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
          <details className="details-box">
            <summary>Why this section is secondary</summary>
            <div className="details-content">
              <p>
                These numbers validate the regime-summary classifier, but the main product
                value of Crystal Forge is the combined workflow: identify the regime, check
                solver trust, then plan measurements.
              </p>
            </div>
          </details>
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
          <div className="endpoint-box">
            <strong>Backend source for these values</strong>
            <code>GET /api/state/export</code>
            <span>
              These numbers are the common language shared by the classifier, CorrMap,
              and QProbe. When anything changes in the backend, these are the first values
              you should cross-check.
            </span>
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
          <details className="details-box">
            <summary>Raw backend snapshot for this page</summary>
            <div className="details-content">
              <pre className="json-box">{prettyJson({
                phase: state.phase,
                observables: state.observables,
                trust: trustResult,
                qprobe: qprobeResult,
                adaptive_qprobe: adaptiveQprobeResult,
              })}</pre>
            </div>
          </details>
        </article>
      </section>
    </main>
  );
}

import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  Box,
  Cpu,
  FolderOpen,
  Gauge,
  Hammer,
  Play,
  Radio,
  Square,
  Terminal,
  Wrench
} from "lucide-react";
import {
  EvaluateArtifact,
  GetState,
  OpenAdvantageScope,
  OpenFolder,
  ReplayArtifact,
  RunAction,
  RunCLI,
  StartSmokeRun,
  StartTraining,
  Stop
} from "../wailsjs/go/main/App";
import "./styles.css";

const defaultConfig = {
  timesteps: "100000",
  modelOut: "models/reefscape_ppo",
  resumeFrom: "",
  device: "cuda",
  robotProfile: "2025-robot",
  preview: "desktop",
  nEnvs: "8",
  nSteps: "512",
  batchSize: "1024",
  learningRate: "0.0003",
  checkpointEvery: "10000",
  pretrain: true,
  advantageScope: true
};

const actions = [
  ["setup_deps", Wrench, "Setup + Deps"],
  ["doctor", Activity, "Doctor"],
  ["checks", Hammer, "Checks"],
  ["build", Cpu, "Build"],
  ["installer", Box, "Installer"]
];

function App() {
  const [state, setState] = useState(null);
  const [tab, setTab] = useState("train");
  const [config, setConfig] = useState(defaultConfig);
  const [cli, setCli] = useState("python menu.py");
  const [selectedModel, setSelectedModel] = useState("");
  const [error, setError] = useState("");

  const refresh = async () => {
    try {
      const next = await GetState();
      setState(next);
      setError("");
    } catch (err) {
      setError(String(err));
    }
  };

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 700);
    return () => window.clearInterval(id);
  }, []);

  const run = async (fn) => {
    try {
      setError("");
      await fn();
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  };

  const latestMetric = state?.metrics?.at(-1);
  const snapshot = state?.snapshot || demoSnapshot();

  return (
    <main className="app">
      <aside className="rail">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <h1>Reefscape RL</h1>
            <p>{state?.running ? state.current : "Systems ready"}</p>
          </div>
        </div>

        <div className="status-card">
          <div className="status-orb" />
          <div>
            <strong>{state?.running ? "Running" : "Ready"}</strong>
            <span>{state?.pythonAvailable ? "Python linked" : "Python missing"}</span>
          </div>
        </div>

        <nav className="tabs">
          <button className={tab === "train" ? "active" : ""} onClick={() => setTab("train")}><Play size={18}/> Train</button>
          <button className={tab === "field" ? "active" : ""} onClick={() => setTab("field")}><Bot size={18}/> Field</button>
          <button className={tab === "models" ? "active" : ""} onClick={() => setTab("models")}><Box size={18}/> Models</button>
          <button className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}><Terminal size={18}/> Logs</button>
        </nav>

        <div className="quick-grid">
          {actions.map(([id, Icon, label]) => (
            <button key={id} className="icon-command" disabled={state?.running} onClick={() => run(() => RunAction(id))}>
              <Icon size={17}/>
              <span>{label}</span>
            </button>
          ))}
        </div>

        <button className="wide-command" onClick={() => run(OpenFolder)}><FolderOpen size={17}/> Project Folder</button>
        <button className="wide-command" onClick={() => run(OpenAdvantageScope)}><Radio size={17}/> AdvantageScope</button>
        <button className="egg" title="2220" />
      </aside>

      <section className="content">
        <header className="hero">
          <div className="hero-copy">
            <span>Blue alliance policy stack</span>
            <h2>Molten Glass Drive Lab</h2>
            <p>Native React desktop control for CUDA training, live field state, models, and telemetry.</p>
          </div>
          <div className="hero-stats">
            <Metric label="Step" value={formatInt(latestMetric?.step || snapshot?.training?.step || 0)} />
            <Metric label="Reward" value={formatMetric(latestMetric?.reward ?? snapshot?.match?.totalReward)} />
            <Metric label="FPS" value={formatMetric(latestMetric?.fps)} />
          </div>
        </header>

        {error && <div className="error">{error}</div>}
        {state?.current && <div className="running-banner">Running: {state.current}</div>}
        {state?.missingTraining?.length > 0 && (
          <div className="warning-banner">
            Training deps missing: {state.missingTraining.join(", ")}. Run Setup + Deps and wait for it to finish.
          </div>
        )}

        {tab === "train" && (
          <TrainView
            config={config}
            setConfig={setConfig}
            running={state?.running}
            onStart={() => run(() => StartTraining(config))}
            onSmoke={() => run(StartSmokeRun)}
            onStop={() => run(Stop)}
            metrics={state?.metrics || []}
            logs={state?.logs || []}
            current={state?.current || ""}
          />
        )}
        {tab === "field" && <FieldView snapshot={snapshot} live={Boolean(state?.snapshot)} />}
        {tab === "models" && (
          <ModelsView
            artifacts={state?.artifacts || []}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            onEvaluate={() => run(() => EvaluateArtifact(selectedModel))}
            onReplay={() => run(() => ReplayArtifact(selectedModel))}
          />
        )}
        {tab === "logs" && (
          <LogsView logs={state?.logs || []} cli={cli} setCli={setCli} onRun={() => run(() => RunCLI(cli))} running={state?.running} />
        )}
      </section>
    </main>
  );
}

function TrainView({ config, setConfig, running, onStart, onSmoke, onStop, metrics, logs, current }) {
  const update = (key, value) => setConfig((current) => ({ ...current, [key]: value }));
  return (
    <div className="grid-layout">
      <section className="panel form-panel">
        <div className="section-title"><Gauge size={18}/> Launch Config</div>
        <div className="form-grid">
          <Field label="Timesteps" value={config.timesteps} onChange={(v) => update("timesteps", v)} />
          <Select label="Device" value={config.device} onChange={(v) => update("device", v)} options={["cuda", "auto", "cpu"]} />
          <Select label="Robot" value={config.robotProfile} onChange={(v) => update("robotProfile", v)} options={["2025-robot", "sim"]} />
          <Select label="Preview" value={config.preview} onChange={(v) => update("preview", v)} options={["desktop", "both", "advantagescope", "none"]} />
          <Field label="Model Out" value={config.modelOut} onChange={(v) => update("modelOut", v)} wide />
          <Field label="Resume From" value={config.resumeFrom} onChange={(v) => update("resumeFrom", v)} wide />
          <Field label="Parallel Envs" value={config.nEnvs} onChange={(v) => update("nEnvs", v)} />
          <Field label="Rollout Steps" value={config.nSteps} onChange={(v) => update("nSteps", v)} />
          <Field label="Batch Size" value={config.batchSize} onChange={(v) => update("batchSize", v)} />
          <Field label="Learning Rate" value={config.learningRate} onChange={(v) => update("learningRate", v)} />
          <Field label="Checkpoint Every" value={config.checkpointEvery} onChange={(v) => update("checkpointEvery", v)} />
        </div>
        <div className="switch-row">
          <label><input type="checkbox" checked={config.pretrain} onChange={(e) => update("pretrain", e.target.checked)} /> Heuristic pretrain</label>
          <label><input type="checkbox" checked={config.advantageScope} onChange={(e) => update("advantageScope", e.target.checked)} /> AdvantageScope stream</label>
        </div>
        <div className="action-row">
          <button className="primary" disabled={running} onClick={onStart}><Play size={18}/> Start Training</button>
          <button disabled={running} onClick={onSmoke}>Smoke Test</button>
          <button className="danger" disabled={!running} onClick={onStop}><Square size={16}/> Stop</button>
        </div>
      </section>

      <section className="panel">
        <div className="section-title"><Activity size={18}/> Current Status</div>
        <TrainingStatus logs={logs} current={current} metrics={metrics} />
      </section>

      <section className="panel wide-panel">
        <div className="section-title"><Activity size={18}/> Metrics</div>
        <MetricChart metrics={metrics} />
        <div className="metric-strip">
          <Metric label="Loss" value={formatMetric(metrics.at(-1)?.loss)} />
          <Metric label="Reward" value={formatMetric(metrics.at(-1)?.reward)} />
          <Metric label="Step" value={formatInt(metrics.at(-1)?.step || 0)} />
        </div>
      </section>
    </div>
  );
}

function TrainingStatus({ logs, current, metrics }) {
  const status = useMemo(() => deriveTrainingStatus(logs, current, metrics), [logs, current, metrics]);
  return (
    <div className="status-flow">
      <div className="now-card">
        <span>Current</span>
        <strong>{status.current}</strong>
        <small>{status.detail}</small>
      </div>
      <div className="phase-list">
        {status.phases.map((phase) => (
          <div key={phase.label} className={`phase ${phase.state}`}>
            <span className="phase-dot" />
            <div>
              <strong>{phase.label}</strong>
              <small>{phase.detail}</small>
            </div>
          </div>
        ))}
      </div>
      <pre className="mini-terminal">{logs.slice(-10).join("\n")}</pre>
    </div>
  );
}

function FieldView({ snapshot, live }) {
  return (
    <section className="panel field-panel">
      <div className="section-title">
        <Bot size={18}/> Native Field
        <span className={live ? "live-pill" : "live-pill idle"}>{live ? "Live" : "Demo"}</span>
      </div>
      <FieldCanvas snapshot={snapshot} />
    </section>
  );
}

function ModelsView({ artifacts, selectedModel, setSelectedModel, onEvaluate, onReplay }) {
  return (
    <section className="panel table-panel">
      <div className="section-title"><Box size={18}/> Models</div>
      <div className="model-actions">
        <button disabled={!selectedModel} onClick={onEvaluate}>Evaluate</button>
        <button disabled={!selectedModel} onClick={onReplay}>Replay</button>
      </div>
      <div className="table">
        {artifacts.map((item) => (
          <button key={item.path} className={selectedModel === item.path ? "row selected" : "row"} onClick={() => setSelectedModel(item.path)}>
            <span>{item.path}</span>
            <small>{item.size}</small>
            <small>{item.modified}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function LogsView({ logs, cli, setCli, onRun, running }) {
  const tail = logs.slice(-700).join("\n");
  return (
    <section className="panel logs-panel">
      <div className="section-title"><Terminal size={18}/> Live Log</div>
      <div className="cli-row">
        <input value={cli} onChange={(e) => setCli(e.target.value)} />
        <button disabled={running} onClick={onRun}>Run CLI</button>
      </div>
      <pre>{tail}</pre>
    </section>
  );
}

function Field({ label, value, onChange, wide }) {
  return (
    <label className={wide ? "field wide" : "field"}>
      <span>{label}</span>
      <input value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function MetricChart({ metrics }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * ratio);
    canvas.height = Math.floor(rect.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    drawChart(ctx, rect.width, rect.height, metrics);
  }, [metrics]);
  return <canvas className="chart" ref={canvasRef} />;
}

function FieldCanvas({ snapshot }) {
  const ref = useRef(null);
  const snapshotRef = useRef(snapshot);
  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let frame = 0;
    const render = (time) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const nextW = Math.floor(rect.width * ratio);
      const nextH = Math.floor(rect.height * ratio);
      if (canvas.width !== nextW || canvas.height !== nextH) {
        canvas.width = nextW;
        canvas.height = nextH;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawField(ctx, rect.width, rect.height, snapshotRef.current, time);
      frame = window.requestAnimationFrame(render);
    };
    frame = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(frame);
  }, []);
  return <canvas className="field-canvas" ref={ref} />;
}

function drawChart(ctx, width, height, metrics) {
  const pad = 26;
  const grd = ctx.createLinearGradient(0, 0, width, height);
  grd.addColorStop(0, "rgba(30, 233, 190, .20)");
  grd.addColorStop(1, "rgba(40, 130, 255, .09)");
  ctx.fillStyle = grd;
  roundRect(ctx, 0, 0, width, height, 22);
  ctx.fill();
  ctx.strokeStyle = "rgba(135,255,220,.18)";
  ctx.lineWidth = 1;
  for (let x = pad; x < width; x += 58) line(ctx, x, pad, x, height - pad);
  for (let y = pad; y < height; y += 42) line(ctx, pad, y, width - pad, y);
  const points = metrics.filter((m) => Number.isFinite(m.reward) && Number.isFinite(m.step));
  if (points.length < 2) {
    ctx.fillStyle = "rgba(220,255,245,.72)";
    ctx.font = "600 15px Inter, Segoe UI, sans-serif";
    ctx.fillText("Waiting for trainer metrics", 28, 42);
    return;
  }
  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.reward);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = pad + ((p.step - minX) / spanX) * (width - pad * 2);
    const y = height - pad - ((p.reward - minY) / spanY) * (height - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#6dffbf";
  ctx.lineWidth = 3;
  ctx.shadowColor = "rgba(109,255,191,.6)";
  ctx.shadowBlur = 14;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawField(ctx, width, height, snap, time = 0) {
  const field = snap.field;
  const pad = 48;
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createRadialGradient(width * .15, height * .15, 20, width * .55, height * .45, width * .8);
  bg.addColorStop(0, "#064e76");
  bg.addColorStop(.45, "#052440");
  bg.addColorStop(1, "#020814");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  const drift = Math.sin(time / 1100) * 18;
  ctx.fillStyle = "rgba(43,255,188,.10)";
  ctx.beginPath(); ctx.ellipse(width * .8 + drift, height * .18, 280, 160, -.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "rgba(50,144,255,.12)";
  ctx.beginPath(); ctx.ellipse(width * .2 - drift, height * .82, 300, 130, .25, 0, Math.PI * 2); ctx.fill();

  const scale = Math.min((width - pad * 2) / field.lengthM, (height - pad * 2 - 96) / field.widthM);
  const ox = (width - field.lengthM * scale) / 2;
  const oy = 44;
  const toPx = (p) => ({ x: ox + p.x * scale, y: oy + (field.widthM - p.y) * scale });
  const fw = field.lengthM * scale;
  const fh = field.widthM * scale;
  glassRect(ctx, ox - 20, oy - 20, fw + 40, fh + 40, 30);
  const fieldGrad = ctx.createLinearGradient(ox, oy, ox + fw, oy + fh);
  fieldGrad.addColorStop(0, "#073d66");
  fieldGrad.addColorStop(.48, "#05364b");
  fieldGrad.addColorStop(1, "#064d3d");
  ctx.fillStyle = fieldGrad;
  roundRect(ctx, ox, oy, fw, fh, 22); ctx.fill();
  ctx.strokeStyle = "rgba(127,255,221,.5)"; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = "rgba(139,255,226,.11)"; ctx.lineWidth = 1;
  for (let x = 1; x < field.lengthM; x += 1) line(ctx, toPx({x, y:0}).x, oy, toPx({x, y:0}).x, oy + fh);
  for (let y = 1; y < field.widthM; y += 1) line(ctx, ox, toPx({x:0, y}).y, ox + fw, toPx({x:0, y}).y);

  const reef = toPx(field.reefCenter);
  ctx.fillStyle = "rgba(92,255,189,.22)";
  ctx.strokeStyle = "#7effcf";
  ctx.lineWidth = 2;
  polygon(ctx, reef.x, reef.y, field.reefObstacleRadiusM * scale, 6, -Math.PI / 6);
  ctx.fill(); ctx.stroke();
  field.reefScoringPoses?.forEach((pose, index) => {
    const p = toPx(pose);
    ctx.strokeStyle = index === snap.objective.goalIndex ? "#77ffbd" : "rgba(124,217,255,.35)";
    ctx.lineWidth = index === snap.objective.goalIndex ? 3 : 1;
    line(ctx, reef.x, reef.y, p.x, p.y);
    dot(ctx, p.x, p.y, index === snap.objective.goalIndex ? 8 : 5, index === snap.objective.goalIndex ? "#77ffbd" : "#74d4ff");
  });
  field.coralStations?.forEach((s) => {
    const p = toPx(s);
    pill(ctx, p.x - 18, p.y - 32, 36, 64, "#6dffbf");
  });
  field.trafficPath?.forEach((p, i, arr) => {
    const a = toPx(p), b = toPx(arr[(i + 1) % arr.length]);
    ctx.strokeStyle = "rgba(106,217,255,.48)";
    ctx.lineWidth = 2;
    line(ctx, a.x, a.y, b.x, b.y);
  });
  const robot = toPx(snap.robot.pose);
  const objective = toPx(snap.objective.pose);
  ctx.strokeStyle = "rgba(109,255,191,.5)";
  ctx.lineWidth = 8;
  line(ctx, robot.x, robot.y, objective.x, objective.y);
  drawRobot(ctx, toPx(snap.traffic.pose), snap.traffic.pose.heading, "#54cfff", "D");
  drawRobot(ctx, robot, snap.robot.pose.heading, "#6dffbf", "AI");
  dot(ctx, objective.x, objective.y, 14 + Math.sin(time / 180) * 2, "#7dffb9");
  const coral = snap.robot.hasCoral ? {x: robot.x + 18, y: robot.y - 18} : toPx(snap.objective.coralPose);
  pill(ctx, coral.x - 14, coral.y - 7, 28, 14, "#9affcf");
  const hudY = height - 96;
  glassRect(ctx, 24, hudY, width - 48, 72, 22);
  ctx.fillStyle = "#e9fff7";
  ctx.font = "700 16px Inter, Segoe UI, sans-serif";
  ctx.fillText(`${snap.running ? "LIVE" : "DEMO"}  ${snap.policy || "desktop"}  step ${snap.training?.step || 0}`, 50, hudY + 28);
  ctx.font = "500 13px Inter, Segoe UI, sans-serif";
  ctx.fillStyle = "rgba(224,255,247,.76)";
  ctx.fillText(`Coral ${snap.match.scoredCoral}/${snap.match.maxCoral}    ${snap.match.timeRemainingS.toFixed(1)}s    ${snap.ai.intent} -> ${snap.ai.focus}`, 50, hudY + 52);
}

function deriveTrainingStatus(logs, current, metrics) {
  const text = logs.join("\n");
  const latest = logs.at(-1) || "";
  const epochMatches = [...text.matchAll(/Heuristic pretrain epoch\s+(\d+)\/(\d+):\s+loss=([0-9.]+)/g)];
  const lastEpoch = epochMatches.at(-1);
  const metric = metrics.at(-1);
  const phases = [
    phase("Environment", /Using device:|CUDA device:/.test(text), current === "Setup + Install Training Dependencies" ? "active" : "pending", firstMatch(text, /Using device:.*|CUDA device:.*/)),
    phase("Heuristic training", Boolean(lastEpoch), /Heuristic pretraining/.test(text) && !/Training metrics will be written/.test(text), lastEpoch ? `epoch ${lastEpoch[1]}/${lastEpoch[2]} loss ${lastEpoch[3]}` : "waiting for imitation pretrain"),
    phase("PPO training", Boolean(metric?.step), current === "Train PPO", metric?.step ? `step ${formatInt(metric.step)} reward ${formatMetric(metric.reward)}` : "waiting for rollout metrics"),
    phase("Visualizer", /visualization enabled|visualizer state/i.test(text), false, firstMatch(text, /Training visualization enabled.*|Native REEFSCAPE visualizer state.*/i)),
    phase("Checkpoint", /Checkpoint saved:|Wrote .*\.zip/.test(text), false, firstMatch(text, /Checkpoint saved:.*|Wrote .*\.zip/))
  ];

  let currentLabel = current || "Idle";
  let detail = latest || "No command running.";
  if (lastEpoch && current === "Train PPO") {
    currentLabel = "Heuristic training";
    detail = `epoch ${lastEpoch[1]}/${lastEpoch[2]} loss ${lastEpoch[3]}`;
  }
  if (metric?.step && current === "Train PPO") {
    currentLabel = "PPO training";
    detail = `step ${formatInt(metric.step)} reward ${formatMetric(metric.reward)} fps ${formatMetric(metric.fps)}`;
  }
  return { current: currentLabel, detail, phases };
}

function phase(label, done, active, detail) {
  return { label, detail: detail || "not started", state: active ? "active" : done ? "done" : "pending" };
}

function firstMatch(text, regex) {
  const match = text.match(regex);
  return match ? match[0] : "";
}

function drawRobot(ctx, p, heading, color, label) {
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(-heading);
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;
  ctx.fillStyle = color;
  roundRect(ctx, -26, -20, 52, 40, 12); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(2,10,18,.72)";
  roundRect(ctx, -18, -13, 36, 26, 8); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.moveTo(24, 0); ctx.lineTo(8, -8); ctx.lineTo(8, 8); ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.fillStyle = "#03101c";
  ctx.font = "800 12px Inter, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, p.x, p.y + 4);
  ctx.textAlign = "left";
}

function demoSnapshot() {
  const reefCenter = { x: 4.48, y: 4.02 };
  return {
    running: false,
    policy: "desktop demo",
    field: {
      lengthM: 17.55,
      widthM: 8.05,
      reefCenter,
      reefObstacleRadiusM: .84,
      robotRadiusM: .46,
      coralStations: [{ x: 1.2, y: 1.1 }, { x: 1.2, y: 6.95 }],
      trafficPath: [{ x: 6.2, y: 1.3 }, { x: 8.8, y: 1.3 }, { x: 8.8, y: 6.7 }, { x: 6.2, y: 6.7 }],
      reefScoringPoses: Array.from({ length: 6 }, (_, i) => ({ x: reefCenter.x + Math.cos(i * Math.PI / 3) * 1.18, y: reefCenter.y + Math.sin(i * Math.PI / 3) * 1.18, heading: 0 }))
    },
    match: { timeRemainingS: 150, scoredCoral: 0, maxCoral: 12, targetLevel: "L4", totalReward: 0 },
    robot: { pose: { x: 2.4, y: 4.1, heading: .1 }, hasCoral: false },
    traffic: { pose: { x: 7.3, y: 2, heading: 1.57 }, distanceM: 5.1 },
    objective: { pose: { x: 4.95, y: 4.83, heading: 0 }, coralPose: { x: 1.2, y: 6.95, heading: 0 }, goalIndex: 1 },
    ai: { intent: "awaiting training", focus: "reef", confidence: .72, risk: .18 },
    training: { step: 0 }
  };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function glassRect(ctx, x, y, w, h, r) {
  const g = ctx.createLinearGradient(x, y, x + w, y + h);
  g.addColorStop(0, "rgba(255,255,255,.14)");
  g.addColorStop(.35, "rgba(45,255,193,.10)");
  g.addColorStop(1, "rgba(48,122,255,.10)");
  ctx.fillStyle = g;
  roundRect(ctx, x, y, w, h, r); ctx.fill();
  ctx.strokeStyle = "rgba(178,255,231,.24)";
  ctx.stroke();
}
function line(ctx, x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
function dot(ctx, x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
function pill(ctx, x, y, w, h, color) { ctx.fillStyle = color; roundRect(ctx, x, y, w, h, Math.min(w, h) / 2); ctx.fill(); }
function polygon(ctx, x, y, r, sides, offset = 0) {
  ctx.beginPath();
  for (let i = 0; i < sides; i++) {
    const a = offset + i * Math.PI * 2 / sides;
    const px = x + Math.cos(a) * r;
    const py = y + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
}
function formatMetric(value) {
  if (!Number.isFinite(Number(value))) return "-";
  const n = Number(value);
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(3);
}
function formatInt(value) {
  return Number(value || 0).toLocaleString();
}

createRoot(document.getElementById("root")).render(<App />);

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.policies import HeuristicCyclePolicy, RandomPolicy
from reefscape_rl.visualizer_snapshot import build_visualizer_snapshot


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8775


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the 2025 REEFSCAPE visualizer.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Read live visualizer snapshots from a training callback JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = (
        FileBackedVisualizerState(args.state_file)
        if args.state_file is not None
        else VisualizerState(policy_name=args.policy, seed=args.seed)
    )
    server = VisualizerServer((args.host, args.port), VisualizerHandler, state)
    url = f"http://{args.host}:{args.port}"
    print(f"REEFSCAPE 2025 Visualizer: {url}")
    print("Press Ctrl+C to stop the visualizer server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping visualizer.")
    finally:
        server.server_close()
    return 0


class VisualizerState:
    def __init__(self, *, policy_name: str, seed: int) -> None:
        self._lock = threading.Lock()
        self.seed = seed
        self.policy_name = policy_name
        self.running = True
        self.speed = 1.0
        self.last_wall_time = time.time()
        self.last_reward = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.75]
        self.env = self._new_env()
        self.policy = self._new_policy()
        self.env.reset(seed=self.seed)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._advance_to_now()
            return build_visualizer_snapshot(
                self.env,
                action=self.last_action,
                reward=self.last_reward,
                policy_name=self.policy_name,
                running=self.running,
            )

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            action = str(payload.get("action", "")).strip()
            if action == "play":
                self.running = True
                self.last_wall_time = time.time()
            elif action == "pause":
                self.running = False
            elif action == "step":
                self._step_once()
            elif action == "reset":
                self.seed = int(payload.get("seed", self.seed))
                self.env = self._new_env()
                self.policy = self._new_policy()
                self.env.reset(seed=self.seed)
                self.last_reward = 0.0
                self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.75]
                self.last_wall_time = time.time()
            elif action == "setPolicy":
                policy_name = str(payload.get("policy", self.policy_name)).strip()
                if policy_name not in {"heuristic", "random"}:
                    raise ValueError("policy must be heuristic or random")
                self.policy_name = policy_name
                self.policy = self._new_policy()
            elif action == "setSpeed":
                self.speed = max(0.1, min(8.0, float(payload.get("speed", self.speed))))
            else:
                raise ValueError(f"unknown action: {action}")

            return build_visualizer_snapshot(
                self.env,
                action=self.last_action,
                reward=self.last_reward,
                policy_name=self.policy_name,
                running=self.running,
            )

    def _advance_to_now(self) -> None:
        now = time.time()
        elapsed = max(0.0, now - self.last_wall_time)
        self.last_wall_time = now
        if not self.running:
            return
        sim_dt = max(0.0, min(0.6, elapsed * self.speed))
        steps = min(12, int(sim_dt / self.env.config.dt_s))
        for _ in range(steps):
            self._step_once()

    def _step_once(self) -> None:
        state = self.env.state
        if state is None:
            self.env.reset(seed=self.seed)
        action = self.policy(self.env)
        _, reward, terminated, truncated, _ = self.env.step(action)
        self.last_action = [float(value) for value in action]
        self.last_reward = float(reward)
        if terminated or truncated:
            self.seed += 1
            self.env.reset(seed=self.seed)

    def _new_env(self) -> ReefscapeEnv:
        return ReefscapeEnv(
            ReefscapeEnvConfig(
                randomize_start=False,
                randomize_other_robot_behavior=True,
                other_robot_enabled=True,
            )
        )

    def _new_policy(self):
        if self.policy_name == "random":
            return RandomPolicy(self.seed)
        return HeuristicCyclePolicy()


class FileBackedVisualizerState:
    def __init__(self, path: Path) -> None:
        self.path = path

    def snapshot(self) -> dict[str, Any]:
        if not self.path.exists():
            raise RuntimeError(f"waiting for training state: {self.path}")
        return json.loads(self.path.read_text(encoding="utf-8"))

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        if action not in {"play", "pause", "step", "reset", "setPolicy", "setSpeed"}:
            raise ValueError(f"unknown action: {action}")
        return self.snapshot()


class VisualizerServer(ThreadingHTTPServer):
    def __init__(self, address, handler, state: VisualizerState):
        super().__init__(address, handler)
        self.state = state


class VisualizerHandler(BaseHTTPRequestHandler):
    server: VisualizerServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send_json(self.server.state.snapshot())
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/control":
                self._send_json(self.server.state.control(payload))
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>REEFSCAPE 2025 Visualizer</title>
  <style>
    :root {
      --bg: #eef2ef;
      --surface: #ffffff;
      --ink: #16211f;
      --muted: #60706b;
      --line: #d8e0da;
      --field: #dfece4;
      --field-dark: #c9dbd1;
      --blue: #2458d4;
      --blue-deep: #15327e;
      --red: #c03f36;
      --red-deep: #84231f;
      --coral: #f28a46;
      --coral-dark: #9d552a;
      --reef: #3d7469;
      --amber: #d7a12e;
      --purple: #6f5aa7;
      --danger: #b73838;
      --ok: #297a48;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 "Segoe UI", system-ui, sans-serif;
      letter-spacing: 0;
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      box-shadow: 0 1px 0 rgba(22,33,31,.04);
    }
    h1 {
      margin: 0;
      font-size: 19px;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: minmax(620px, 1fr) 360px;
      min-height: calc(100vh - 58px);
    }
    .fieldBand {
      padding: 18px;
      min-width: 0;
    }
    .canvasShell {
      width: 100%;
      height: calc(100vh - 96px);
      min-height: 520px;
      border: 1px solid #c9d5ce;
      background: #f9fbfa;
      box-shadow: 0 12px 28px rgba(30, 48, 42, .10);
      overflow: hidden;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--surface);
      padding: 18px;
      overflow-y: auto;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    button, select, input {
      height: 34px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      min-width: 38px;
    }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    button:hover, select:hover {
      border-color: #aebbb4;
    }
    button.icon {
      width: 36px;
      padding: 0;
      font-size: 16px;
    }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 16px;
      margin-top: 16px;
    }
    .section:first-child {
      border-top: 0;
      padding-top: 0;
      margin-top: 0;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .kpi {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 66px;
      background: #fbfcfb;
    }
    .kpi strong {
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }
    .kpi span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .meter {
      height: 8px;
      background: #edf1ee;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }
    .meter > div {
      height: 100%;
      background: var(--ok);
      width: 0%;
    }
    .meter.risk > div { background: var(--danger); }
    .trace {
      min-height: 52px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      background: #fbfcfb;
      font-size: 13px;
    }
    .legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      color: var(--muted);
    }
    .legendItem {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,.1);
    }
    .stack {
      display: grid;
      gap: 10px;
    }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
    }
    .row strong { color: var(--ink); }
    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }
      aside {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
      .canvasShell {
        height: 62vh;
        min-height: 380px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>REEFSCAPE 2025 Visualizer</h1>
    <div class="toolbar">
      <button id="playButton" class="primary icon" title="Play or pause">||</button>
      <button id="stepButton" class="icon" title="Step one tick">>|</button>
      <button id="resetButton" class="icon" title="Reset simulation">R</button>
      <select id="policySelect" title="Policy">
        <option value="training model">Training model</option>
        <option value="heuristic">Heuristic</option>
        <option value="random">Random</option>
      </select>
      <input id="speedInput" type="range" min="0.1" max="8" step="0.1" value="1" title="Simulation speed">
      <span id="speedLabel">1.0x</span>
    </div>
  </header>
  <main>
    <section class="fieldBand">
      <div class="canvasShell">
        <canvas id="fieldCanvas"></canvas>
      </div>
    </section>
    <aside>
      <section class="section">
        <h2>Match</h2>
        <div class="kpis">
          <div class="kpi"><strong id="scoreKpi">0/12</strong><span>coral scored</span></div>
          <div class="kpi"><strong id="timeKpi">150.0s</strong><span>time remaining</span></div>
          <div class="kpi"><strong id="levelKpi">L4</strong><span>target level</span></div>
          <div class="kpi"><strong id="rewardKpi">0.00</strong><span>total reward</span></div>
        </div>
      </section>
      <section class="section">
        <h2>AI Driver</h2>
        <div class="stack">
          <div class="row"><span>Intent</span><strong id="intentText">-</strong></div>
          <div class="row"><span>Focus</span><strong id="focusText">-</strong></div>
          <div>
            <div class="row"><span>Confidence</span><strong id="confidenceText">0%</strong></div>
            <div class="meter"><div id="confidenceMeter"></div></div>
          </div>
          <div>
            <div class="row"><span>Traffic risk</span><strong id="riskText">0%</strong></div>
            <div class="meter risk"><div id="riskMeter"></div></div>
          </div>
          <div class="trace" id="traceText">Waiting for simulator state.</div>
        </div>
      </section>
      <section class="section">
        <h2>Robot State</h2>
        <div class="stack">
          <div class="row"><span>Coral</span><strong id="coralText">-</strong></div>
          <div class="row"><span>Objective distance</span><strong id="objectiveText">-</strong></div>
          <div class="row"><span>Traffic distance</span><strong id="trafficText">-</strong></div>
          <div class="row"><span>Contact</span><strong id="contactText">-</strong></div>
          <div class="row"><span>Action</span><strong id="actionText">-</strong></div>
        </div>
      </section>
      <section class="section">
        <h2>2025 Overlays</h2>
        <div class="legend">
          <div class="legendItem"><span class="swatch" style="background:var(--blue)"></span>AI robot</div>
          <div class="legendItem"><span class="swatch" style="background:var(--red)"></span>Traffic robot</div>
          <div class="legendItem"><span class="swatch" style="background:var(--reef)"></span>Reef</div>
          <div class="legendItem"><span class="swatch" style="background:var(--coral)"></span>Coral/source</div>
          <div class="legendItem"><span class="swatch" style="background:var(--amber)"></span>Objective</div>
          <div class="legendItem"><span class="swatch" style="background:var(--purple)"></span>AI focus</div>
        </div>
      </section>
    </aside>
  </main>
  <script>
    const canvas = document.getElementById("fieldCanvas");
    const ctx = canvas.getContext("2d");
    let latest = null;
    let running = true;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function refresh() {
      latest = await api("/api/state");
      running = latest.running;
      renderHud(latest);
      draw();
      window.setTimeout(() => refresh().catch(showError), 100);
    }

    async function control(action, extra = {}) {
      latest = await api("/api/control", {
        method: "POST",
        body: JSON.stringify({action, ...extra})
      });
      running = latest.running;
      renderHud(latest);
      draw();
    }

    function renderHud(data) {
      document.getElementById("playButton").textContent = data.running ? "||" : ">";
      document.getElementById("policySelect").value = data.policy;
      document.getElementById("scoreKpi").textContent = `${data.match.scoredCoral}/${data.match.maxCoral}`;
      document.getElementById("timeKpi").textContent = data.match.timeRemainingS.toFixed(1) + "s";
      document.getElementById("levelKpi").textContent = data.match.targetLevel;
      document.getElementById("rewardKpi").textContent = data.match.totalReward.toFixed(2);
      document.getElementById("intentText").textContent = data.ai.intent;
      document.getElementById("focusText").textContent = data.ai.focus;
      document.getElementById("confidenceText").textContent = pct(data.ai.confidence);
      document.getElementById("riskText").textContent = pct(data.ai.risk);
      document.getElementById("confidenceMeter").style.width = pct(data.ai.confidence);
      document.getElementById("riskMeter").style.width = pct(data.ai.risk);
      document.getElementById("traceText").textContent = data.ai.thoughtTrace;
      document.getElementById("coralText").textContent = data.robot.hasCoral ? "carried" : "at source";
      document.getElementById("objectiveText").textContent = data.objective.distanceM.toFixed(2) + " m";
      document.getElementById("trafficText").textContent = data.traffic.distanceM.toFixed(2) + " m";
      document.getElementById("contactText").textContent = data.traffic.hardHit ? "hard hit" : data.traffic.hit ? "tap" : "clear";
      document.getElementById("actionText").textContent = data.ai.action.slice(0, 3).map((v) => v.toFixed(2)).join(", ");
    }

    function draw() {
      if (!latest) return;
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const field = latest.field;
      const pad = 26;
      const scale = Math.min((w - pad * 2) / field.lengthM, (h - pad * 2) / field.widthM);
      const ox = (w - field.lengthM * scale) / 2;
      const oy = (h - field.widthM * scale) / 2;
      const toPx = (p) => ({x: ox + p.x * scale, y: oy + (field.widthM - p.y) * scale});

      drawField(field, toPx, scale);
      drawTrafficPath(field, toPx);
      drawVelocityTrail(latest.traffic.pose, latest.traffic.vxMps, latest.traffic.vyMps, "#c03f36", toPx, scale);
      drawVelocityTrail(latest.robot.pose, latest.robot.vxMps, latest.robot.vyMps, "#2458d4", toPx, scale);
      drawObjective(latest, toPx, scale);
      drawReef(field, toPx, scale);
      drawStations(field, toPx, scale);
      drawRobot(latest.traffic.pose, field.robotRadiusM, "#c03f36", "#84231f", "D", toPx, scale, latest.traffic.hardHit);
      drawRobot(latest.robot.pose, field.robotRadiusM, "#2458d4", "#15327e", "AI", toPx, scale, latest.traffic.hit);
      drawCoral(latest.objective.coralPose, latest.robot.hasCoral, toPx, scale);
      drawAttention(latest.ai.attentionPose, toPx, scale);
      drawFieldLabels(field, toPx);
    }

    function drawField(field, toPx, scale) {
      const topLeft = toPx({x: 0, y: field.widthM});
      const bottomRight = toPx({x: field.lengthM, y: 0});
      const fieldWidth = bottomRight.x - topLeft.x;
      const fieldHeight = bottomRight.y - topLeft.y;
      ctx.fillStyle = "#dfece4";
      ctx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
      ctx.fillStyle = "rgba(255,255,255,.24)";
      for (let band = 0; band < 8; band += 1) {
        if (band % 2 === 0) {
          ctx.fillRect(topLeft.x, topLeft.y + (fieldHeight / 8) * band, fieldWidth, fieldHeight / 8);
        }
      }
      ctx.strokeStyle = "#8da397";
      ctx.lineWidth = 2;
      ctx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
      ctx.fillStyle = "rgba(36,88,212,.10)";
      ctx.fillRect(topLeft.x, topLeft.y, (bottomRight.x - topLeft.x) / 2, bottomRight.y - topLeft.y);
      ctx.fillStyle = "rgba(192,63,54,.07)";
      ctx.fillRect(topLeft.x + fieldWidth / 2, topLeft.y, fieldWidth / 2, fieldHeight);
      const midA = toPx({x: field.lengthM / 2, y: 0});
      const midB = toPx({x: field.lengthM / 2, y: field.widthM});
      ctx.strokeStyle = "rgba(22,33,31,.34)";
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 8]);
      ctx.beginPath(); ctx.moveTo(midA.x, midA.y); ctx.lineTo(midB.x, midB.y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = "rgba(22,33,31,.18)";
      ctx.lineWidth = 1;
      for (let x = 1; x < field.lengthM; x += 1) {
        const a = toPx({x, y: 0});
        const b = toPx({x, y: field.widthM});
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let y = 1; y < field.widthM; y += 1) {
        const a = toPx({x: 0, y});
        const b = toPx({x: field.lengthM, y});
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      drawWall(topLeft.x, topLeft.y, fieldWidth, fieldHeight);
    }

    function drawWall(x, y, width, height) {
      ctx.fillStyle = "#273a35";
      ctx.fillRect(x - 6, y - 6, width + 12, 6);
      ctx.fillRect(x - 6, y + height, width + 12, 6);
      ctx.fillRect(x - 6, y - 6, 6, height + 12);
      ctx.fillRect(x + width, y - 6, 6, height + 12);
      ctx.fillStyle = "#2458d4";
      ctx.fillRect(x - 6, y - 6, width * 0.5 + 6, 6);
      ctx.fillRect(x - 6, y + height, width * 0.5 + 6, 6);
      ctx.fillStyle = "#c03f36";
      ctx.fillRect(x + width * 0.5, y - 6, width * 0.5 + 6, 6);
      ctx.fillRect(x + width * 0.5, y + height, width * 0.5 + 6, 6);
    }

    function drawReef(field, toPx, scale) {
      const c = toPx(field.reefCenter);
      const activeGoal = latest.objective.goalIndex;
      ctx.save();
      ctx.shadowColor = "rgba(61,116,105,.28)";
      ctx.shadowBlur = 12;
      ctx.beginPath();
      for (let i = 0; i < 6; i += 1) {
        const a = -Math.PI / 6 + i * Math.PI / 3;
        const x = c.x + Math.cos(a) * field.reefObstacleRadiusM * scale;
        const y = c.y - Math.sin(a) * field.reefObstacleRadiusM * scale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(61,116,105,.24)";
      ctx.strokeStyle = "#3d7469";
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      field.reefScoringPoses.forEach((pose, index) => {
        const p = toPx(pose);
        ctx.strokeStyle = index === activeGoal ? "#d7a12e" : "rgba(61,116,105,.45)";
        ctx.lineWidth = index === activeGoal ? 3 : 1.5;
        ctx.beginPath();
        ctx.moveTo(c.x, c.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
      });
      field.reefScoringPoses.forEach((pose, index) => {
        const p = toPx(pose);
        const active = index === activeGoal;
        ctx.beginPath();
        ctx.arc(p.x, p.y, active ? 8 : 5, 0, Math.PI * 2);
        ctx.fillStyle = active ? "#d7a12e" : "#ffffff";
        ctx.fill();
        ctx.strokeStyle = active ? "#7f5a12" : "#3d7469";
        ctx.lineWidth = active ? 2 : 1;
        ctx.stroke();
        if (active) {
          ctx.fillStyle = "#16211f";
          ctx.font = "700 11px Segoe UI, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(latest.match.targetLevel, p.x, p.y - 13);
          ctx.textAlign = "left";
        }
      });
    }

    function drawStations(field, toPx, scale) {
      field.coralStations.forEach((station, index) => {
        const p = toPx(station);
        const active = index === latest.objective.sourceIndex && !latest.robot.hasCoral;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.fillStyle = active ? "#f28a46" : "#f6b27e";
        ctx.strokeStyle = active ? "#7f3d15" : "#9d552a";
        ctx.lineWidth = active ? 3 : 1.5;
        roundRect(ctx, -18, -26, 36, 52, 5);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "rgba(255,255,255,.48)";
        for (let i = -1; i <= 1; i += 1) {
          ctx.beginPath();
          ctx.arc(i * 8, -4, 4, 0, Math.PI * 2);
          ctx.arc(i * 8, 8, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      });
    }

    function drawTrafficPath(field, toPx) {
      ctx.beginPath();
      field.trafficPath.forEach((point, index) => {
        const p = toPx(point);
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.strokeStyle = "rgba(192,63,54,.45)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawObjective(data, toPx, scale) {
      const robot = toPx(data.robot.pose);
      const objective = toPx(data.objective.pose);
      ctx.strokeStyle = "rgba(215,161,46,.32)";
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.moveTo(robot.x, robot.y);
      ctx.lineTo(objective.x, objective.y);
      ctx.stroke();
      ctx.strokeStyle = "rgba(127,90,18,.88)";
      ctx.lineWidth = 2;
      ctx.setLineDash([12, 7]);
      ctx.beginPath();
      ctx.moveTo(robot.x, robot.y);
      ctx.lineTo(objective.x, objective.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(objective.x, objective.y, 13, 0, Math.PI * 2);
      ctx.fillStyle = "#d7a12e";
      ctx.fill();
      ctx.strokeStyle = "#7f5a12";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function drawCoral(pose, carried, toPx, scale) {
      if (carried) {
        const robot = toPx(latest.robot.pose);
        drawCoralPiece(robot.x + 12, robot.y - 12, 8);
        return;
      }
      const p = toPx(pose);
      drawCoralPiece(p.x, p.y, 8);
    }

    function drawCoralPiece(x, y, size) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(-0.55);
      ctx.fillStyle = "#f28a46";
      ctx.strokeStyle = "#8f4a22";
      ctx.lineWidth = 1.5;
      roundRect(ctx, -size * 1.2, -size * .55, size * 2.4, size * 1.1, 4);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function drawVelocityTrail(pose, vx, vy, color, toPx, scale) {
      const speed = Math.hypot(vx, vy);
      if (speed < 0.08) return;
      const p = toPx(pose);
      const len = Math.min(58, speed * 12);
      const angle = Math.atan2(-vy, vx);
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(angle);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-len, 0);
      ctx.lineTo(-12, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-12, 0);
      ctx.lineTo(-22, -6);
      ctx.lineTo(-22, 6);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }

    function drawAttention(pose, toPx, scale) {
      const p = toPx(pose);
      ctx.beginPath();
      const pulse = 18 + Math.sin(Date.now() / 180) * 3;
      ctx.arc(p.x, p.y, pulse, 0, Math.PI * 2);
      ctx.strokeStyle = "#6f5aa7";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x - 22, p.y);
      ctx.lineTo(p.x + 22, p.y);
      ctx.moveTo(p.x, p.y - 22);
      ctx.lineTo(p.x, p.y + 22);
      ctx.strokeStyle = "rgba(111,90,167,.65)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    function drawRobot(pose, radiusM, color, darkColor, label, toPx, scale, alert) {
      const p = toPx(pose);
      const r = Math.max(14, radiusM * scale);
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(-pose.heading);
      if (alert) {
        ctx.shadowColor = "rgba(183,56,56,.62)";
        ctx.shadowBlur = 18;
      }
      ctx.fillStyle = darkColor;
      roundRect(ctx, -r * 1.08, -r * 0.88, r * 2.16, r * 1.76, 7);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.strokeStyle = "#16211f";
      ctx.lineWidth = 2;
      roundRect(ctx, -r * .86, -r * 0.66, r * 1.72, r * 1.32, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.moveTo(r * 0.82, 0);
      ctx.lineTo(r * 0.25, -r * 0.30);
      ctx.lineTo(r * 0.25, r * 0.30);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,.30)";
      ctx.fillRect(-r * .55, -r * .40, r * .28, r * .80);
      ctx.fillRect(-r * .08, -r * .40, r * .28, r * .80);
      ctx.restore();
      ctx.fillStyle = "#ffffff";
      ctx.font = "700 12px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, p.x, p.y);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }

    function drawFieldLabels(field, toPx) {
      const left = toPx({x: 0.35, y: field.widthM - 0.35});
      const right = toPx({x: field.lengthM - 2.15, y: field.widthM - 0.35});
      ctx.fillStyle = "rgba(22,33,31,.65)";
      ctx.font = "700 12px Segoe UI, sans-serif";
      ctx.fillText("BLUE ALLIANCE", left.x, left.y);
      ctx.fillText("FULL FIELD SCALE", right.x, right.y);
    }

    function roundRect(context, x, y, w, h, r) {
      context.beginPath();
      context.moveTo(x + r, y);
      context.lineTo(x + w - r, y);
      context.quadraticCurveTo(x + w, y, x + w, y + r);
      context.lineTo(x + w, y + h - r);
      context.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      context.lineTo(x + r, y + h);
      context.quadraticCurveTo(x, y + h, x, y + h - r);
      context.lineTo(x, y + r);
      context.quadraticCurveTo(x, y, x + r, y);
    }

    function pct(value) {
      return Math.round(Math.max(0, Math.min(1, Number(value || 0))) * 100) + "%";
    }

    document.getElementById("playButton").addEventListener("click", () => control(running ? "pause" : "play"));
    document.getElementById("stepButton").addEventListener("click", () => control("step"));
    document.getElementById("resetButton").addEventListener("click", () => control("reset"));
    document.getElementById("policySelect").addEventListener("change", (event) => control("setPolicy", {policy: event.target.value}));
    document.getElementById("speedInput").addEventListener("input", (event) => {
      const speed = Number(event.target.value);
      document.getElementById("speedLabel").textContent = speed.toFixed(1) + "x";
      control("setSpeed", {speed}).catch(console.error);
    });
    window.addEventListener("resize", draw);
    function showError(error) {
      document.getElementById("traceText").textContent = error.message;
    }
    refresh().catch(showError);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

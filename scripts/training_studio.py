from __future__ import annotations

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_CONFIG: dict[str, Any] = {
    "timesteps": 100_000,
    "modelOut": "models/reefscape_ppo",
    "resumeFrom": "",
    "device": "cuda",
    "nEnvs": 8,
    "nSteps": 512,
    "batchSize": 1024,
    "learningRate": 0.0003,
    "checkpointDir": "models/checkpoints",
    "checkpointEverySteps": 10_000,
    "keepCheckpoints": 2,
    "heuristicPretrain": True,
    "pretrainSamples": 50_000,
    "pretrainEpochs": 10,
    "advantageScope": True,
    "advantagePort": 5810,
    "vizEverySteps": 512,
    "vizPreviewSteps": 25,
    "variedDefense": True,
    "metricsOut": "logs/training_studio_metrics.jsonl",
    "metricsEverySteps": 512,
}

INT_FIELDS = {
    "timesteps",
    "nEnvs",
    "nSteps",
    "batchSize",
    "checkpointEverySteps",
    "keepCheckpoints",
    "pretrainSamples",
    "pretrainEpochs",
    "advantagePort",
    "vizEverySteps",
    "vizPreviewSteps",
    "metricsEverySteps",
}
FLOAT_FIELDS = {"learningRate"}
BOOL_FIELDS = {"heuristicPretrain", "advantageScope", "variedDefense"}
STRING_FIELDS = {"modelOut", "resumeFrom", "device", "checkpointDir", "metricsOut"}
DEVICE_CHOICES = {"auto", "cuda", "cpu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the REEFSCAPE training studio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = TrainingStudioState()
    server = TrainingStudioServer((args.host, args.port), TrainingStudioHandler, state)
    url = f"http://{args.host}:{args.port}"
    print(f"REEFSCAPE Training Studio: {url}")
    print("Press Ctrl+C to stop the studio server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping studio.")
    finally:
        state.stop_training()
        server.server_close()
    return 0


def normalize_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULT_CONFIG)
    if payload:
        data.update(payload)

    config: dict[str, Any] = {}
    for key, default in DEFAULT_CONFIG.items():
        value = data.get(key, default)
        if key in INT_FIELDS:
            config[key] = _coerce_int(key, value)
        elif key in FLOAT_FIELDS:
            config[key] = _coerce_float(key, value)
        elif key in BOOL_FIELDS:
            config[key] = bool(value)
        elif key in STRING_FIELDS:
            config[key] = str(value).strip()
        else:
            config[key] = value

    if config["device"] not in DEVICE_CHOICES:
        raise ValueError("device must be auto, cuda, or cpu")
    if config["timesteps"] < 0:
        raise ValueError("timesteps must be >= 0")
    for key in (
        "nEnvs",
        "nSteps",
        "batchSize",
        "keepCheckpoints",
        "pretrainSamples",
        "pretrainEpochs",
        "vizEverySteps",
        "vizPreviewSteps",
        "metricsEverySteps",
    ):
        if config[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    if config["checkpointEverySteps"] < 0:
        raise ValueError("checkpointEverySteps must be >= 0")
    if config["learningRate"] <= 0:
        raise ValueError("learningRate must be > 0")
    if not config["modelOut"]:
        raise ValueError("modelOut is required")
    if not config["checkpointDir"]:
        raise ValueError("checkpointDir is required")
    if not config["metricsOut"]:
        config["metricsOut"] = _default_metrics_path()

    return config


def validate_launch_config(
    config: dict[str, Any],
    compute_status: dict[str, Any] | None = None,
) -> None:
    status = compute_status or get_compute_status()
    if config["device"] == "cuda" and not status["cudaAvailable"]:
        torch_version = status.get("torchVersion") or "not installed"
        raise ValueError(
            "CUDA was selected, but this Python environment cannot see CUDA. "
            f"torch={torch_version}. Use device=auto/cpu or install a CUDA-enabled "
            "PyTorch build."
        )


def get_compute_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {
            "torchInstalled": False,
            "torchVersion": "",
            "cudaAvailable": False,
            "cudaDeviceCount": 0,
            "cudaDeviceName": "",
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    return {
        "torchInstalled": True,
        "torchVersion": str(torch.__version__),
        "cudaAvailable": cuda_available,
        "cudaDeviceCount": device_count,
        "cudaDeviceName": device_name,
        "error": "",
    }


def build_train_command(config: dict[str, Any]) -> list[str]:
    cmd = [
        PYTHON,
        "scripts/train_ppo.py",
        "--timesteps",
        str(config["timesteps"]),
        "--model-out",
        config["modelOut"],
        "--device",
        config["device"],
        "--n-envs",
        str(config["nEnvs"]),
        "--n-steps",
        str(config["nSteps"]),
        "--batch-size",
        str(config["batchSize"]),
        "--learning-rate",
        str(config["learningRate"]),
        "--checkpoint-dir",
        config["checkpointDir"],
        "--checkpoint-every-steps",
        str(config["checkpointEverySteps"]),
        "--keep-checkpoints",
        str(config["keepCheckpoints"]),
        "--pretrain-heuristic-samples",
        str(config["pretrainSamples"]),
        "--pretrain-heuristic-epochs",
        str(config["pretrainEpochs"]),
        "--advantage-port",
        str(config["advantagePort"]),
        "--viz-every-steps",
        str(config["vizEverySteps"]),
        "--viz-preview-steps",
        str(config["vizPreviewSteps"]),
        "--metrics-out",
        config["metricsOut"],
        "--metrics-every-steps",
        str(config["metricsEverySteps"]),
    ]
    if config["resumeFrom"]:
        cmd.extend(["--resume-from", config["resumeFrom"]])
    if not config["heuristicPretrain"]:
        cmd.append("--skip-heuristic-pretrain")
    if not config["advantageScope"]:
        cmd.append("--no-advantagescope")
    if not config["variedDefense"]:
        cmd.append("--fixed-defense")
    return cmd


class TrainingStudioState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._command: list[str] = []
        self._logs: deque[str] = deque(maxlen=2_000)
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._return_code: int | None = None
        self._last_config = dict(DEFAULT_CONFIG)
        self._metrics_path: Path | None = _resolve_repo_path(
            DEFAULT_CONFIG["metricsOut"]
        )

    def start_training(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        config = normalize_config(payload)
        validate_launch_config(config)
        cmd = build_train_command(config)
        metrics_path = _resolve_repo_path(config["metricsOut"])
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("training is already running")
            self._logs.clear()
            self._logs.append("$ " + " ".join(cmd))
            self._command = cmd
            self._started_at = time.time()
            self._finished_at = None
            self._return_code = None
            self._last_config = config
            self._metrics_path = metrics_path
            self._process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            process = self._process

        threading.Thread(
            target=self._read_output,
            args=(process,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._monitor_process,
            args=(process,),
            daemon=True,
        ).start()
        return self.status()

    def stop_training(self) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            candidate = self._process
            if candidate is not None and candidate.poll() is None:
                process = candidate
                self._logs.append("Stopping training...")

        if process is None:
            return self.status()
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                pass
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            return_code = process.poll() if process is not None else self._return_code
            running = process is not None and return_code is None
            if process is not None and return_code is not None:
                self._return_code = int(return_code)
            uptime_s = (
                time.time() - self._started_at
                if self._started_at is not None and running
                else 0.0
            )
            metrics_path = self._metrics_path
            payload = {
                "running": running,
                "pid": process.pid if process is not None and running else None,
                "returnCode": self._return_code,
                "command": self._command,
                "logs": list(self._logs),
                "startedAt": self._started_at,
                "finishedAt": self._finished_at,
                "uptimeS": uptime_s,
                "config": self._last_config,
                "metricsPath": str(metrics_path) if metrics_path is not None else "",
            }

        payload["compute"] = get_compute_status()
        payload["metrics"] = _read_metrics(metrics_path)
        payload["artifacts"] = _list_artifacts()
        return payload

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            if clean:
                with self._lock:
                    self._logs.append(clean)

    def _monitor_process(self, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        with self._lock:
            if self._process is process:
                self._return_code = int(return_code)
                self._finished_at = time.time()
                self._logs.append(f"Training process exited with code {return_code}.")


class TrainingStudioServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        state: TrainingStudioState,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.state = state


class TrainingStudioHandler(BaseHTTPRequestHandler):
    server: TrainingStudioServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/api/defaults":
            self._send_json({"config": DEFAULT_CONFIG, "compute": get_compute_status()})
        elif path == "/api/status":
            self._send_json(self.server.state.status())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/start":
                self._send_json(self.server.state.start_training(payload))
            elif path == "/api/stop":
                self._send_json(self.server.state.stop_training())
            else:
                self.send_error(404)
        except (RuntimeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _coerce_int(key: str, value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _coerce_float(key: str, value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def _default_metrics_path() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"logs/training_studio_metrics_{stamp}.jsonl"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _read_metrics(path: Path | None, *, limit: int = 1_000) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        start = max(0, size - 2_000_000)
        file.seek(start)
        text = file.read().decode("utf-8", errors="replace")
    lines = text.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    lines = lines[-limit:]
    metrics: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            metrics.append(row)
    return metrics


def _list_artifacts() -> list[dict[str, Any]]:
    model_root = REPO_ROOT / "models"
    if not model_root.exists():
        return []
    artifacts = []
    for path in model_root.rglob("*.zip"):
        try:
            stat = path.stat()
        except OSError:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sizeBytes": stat.st_size,
                "modifiedAt": stat.st_mtime,
            }
        )
    artifacts.sort(key=lambda item: item["modifiedAt"], reverse=True)
    return artifacts[:20]


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>REEFSCAPE Training Studio</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f6;
      --panel: #ffffff;
      --panel-soft: #eef4f0;
      --text: #1d2623;
      --muted: #66736d;
      --line: #d9e1dc;
      --accent: #2f7d5c;
      --accent-strong: #236348;
      --warn: #b77b1d;
      --danger: #b5473f;
      --ink: #22302b;
      --shadow: 0 10px 28px rgba(31, 45, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font: 14px/1.45 "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    button, input, select {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 12px;
      color: var(--ink);
      background: #ffffff;
      cursor: pointer;
    }
    button:hover { border-color: #aebbb4; }
    button.primary {
      color: #ffffff;
      border-color: var(--accent);
      background: var(--accent);
    }
    button.primary:hover { background: var(--accent-strong); }
    button.danger {
      color: #ffffff;
      border-color: var(--danger);
      background: var(--danger);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .app {
      display: grid;
      grid-template-columns: 236px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfb;
      padding: 18px 16px;
    }
    .brand {
      display: grid;
      gap: 2px;
      margin-bottom: 20px;
    }
    .brand strong {
      font-size: 18px;
      letter-spacing: 0;
    }
    .brand span {
      color: var(--muted);
      font-size: 12px;
    }
    .statusBox {
      display: grid;
      gap: 12px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .statusRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .pill {
      min-width: 72px;
      border-radius: 999px;
      padding: 4px 9px;
      text-align: center;
      color: #ffffff;
      background: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .pill.running { background: var(--accent); }
    .pill.stopped { background: var(--muted); }
    .sideActions {
      display: grid;
      gap: 8px;
    }
    main {
      min-width: 0;
      padding: 18px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .topbar h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .topbar .meta {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(340px, 520px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panelHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }
    .panelHeader h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .settings {
      display: grid;
      gap: 16px;
      padding: 14px;
    }
    fieldset {
      display: grid;
      gap: 10px;
      min-width: 0;
      margin: 0;
      padding: 0 0 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
    }
    fieldset:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    legend {
      margin-bottom: 8px;
      color: var(--accent-strong);
      font-weight: 700;
    }
    .formGrid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 9px;
      color: var(--text);
      background: #ffffff;
    }
    input:focus, select:focus {
      outline: 2px solid rgba(47, 125, 92, 0.22);
      border-color: var(--accent);
    }
    .wide { grid-column: 1 / -1; }
    .switches {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfb;
      color: var(--text);
      font-size: 12px;
    }
    .check input {
      width: 16px;
      height: 16px;
      min-width: 16px;
      padding: 0;
    }
    .charts {
      display: grid;
      gap: 14px;
    }
    canvas {
      width: 900px;
      height: 260px;
      display: block;
      background: #ffffff;
    }
    .canvasWrap {
      padding: 8px 10px 12px;
    }
    .chartScroll {
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 12px 14px;
    }
    .kpi {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: var(--panel-soft);
    }
    .kpi span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .kpi strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      margin-top: 3px;
      font-size: 15px;
    }
    .lower {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr);
      gap: 14px;
    }
    pre {
      height: 260px;
      margin: 0;
      padding: 12px;
      overflow: auto;
      color: #e9f1ed;
      background: #17221d;
      font: 12px/1.45 Consolas, "Cascadia Mono", monospace;
      white-space: pre-wrap;
    }
    .artifacts {
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow: auto;
      padding: 10px;
    }
    .artifact {
      display: grid;
      gap: 2px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfb;
      font-size: 12px;
    }
    .artifact strong {
      overflow-wrap: anywhere;
    }
    .artifact span {
      color: var(--muted);
    }
    .error {
      display: none;
      margin: 0 0 14px;
      padding: 10px 12px;
      border: 1px solid #e7c5c1;
      border-radius: 8px;
      color: #7e2b25;
      background: #fff1ef;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .layout, .lower { grid-template-columns: 1fr; }
      .switches, .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 620px) {
      main { padding: 12px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .topbar .meta { text-align: left; }
      .formGrid, .switches, .kpis { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <strong>REEFSCAPE Studio</strong>
        <span>PPO training control</span>
      </div>
      <div class="statusBox">
        <div class="statusRow">
          <span>Status</span>
          <span id="statusPill" class="pill stopped">Idle</span>
        </div>
        <div class="statusRow">
          <span>PID</span>
          <strong id="pidValue">-</strong>
        </div>
        <div class="statusRow">
          <span>Runtime</span>
          <strong id="runtimeValue">0s</strong>
        </div>
        <div class="statusRow">
          <span>Compute</span>
          <strong id="computeValue">Checking</strong>
        </div>
        <div class="sideActions">
          <button id="startButton" class="primary" type="button">Start</button>
          <button id="stopButton" class="danger" type="button" disabled>Stop</button>
        </div>
      </div>
    </aside>
    <main>
      <div class="topbar">
        <h1>Training Run</h1>
        <div class="meta">
          <div id="metricsPath">metrics: -</div>
          <div id="commandLine">command: -</div>
        </div>
      </div>
      <p id="errorBox" class="error"></p>
      <div class="layout">
        <section class="panel">
          <div class="panelHeader">
            <h2>Settings</h2>
            <button id="resetButton" type="button">Reset</button>
          </div>
          <form id="settingsForm" class="settings">
            <fieldset>
              <legend>Run</legend>
              <div class="formGrid">
                <label>Timesteps
                  <input data-key="timesteps" type="number" min="0" step="1">
                </label>
                <label>Device
                  <select data-key="device">
                    <option value="auto">auto</option>
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </select>
                </label>
                <label class="wide">Model output
                  <input data-key="modelOut" type="text">
                </label>
                <label class="wide">Resume from
                  <input data-key="resumeFrom" type="text">
                </label>
              </div>
            </fieldset>
            <fieldset>
              <legend>PPO</legend>
              <div class="formGrid">
                <label>Parallel envs
                  <input data-key="nEnvs" type="number" min="1" step="1">
                </label>
                <label>Rollout steps
                  <input data-key="nSteps" type="number" min="1" step="1">
                </label>
                <label>Batch size
                  <input data-key="batchSize" type="number" min="1" step="1">
                </label>
                <label>Learning rate
                  <input data-key="learningRate" type="number" min="0" step="0.00001">
                </label>
              </div>
            </fieldset>
            <fieldset>
              <legend>Pretrain and defense</legend>
              <div class="switches">
                <label class="check">
                  <input data-key="heuristicPretrain" type="checkbox">
                  Heuristic pretrain
                </label>
                <label class="check">
                  <input data-key="variedDefense" type="checkbox">
                  Varied defense
                </label>
                <label class="check">
                  <input data-key="advantageScope" type="checkbox">
                  AdvantageScope
                </label>
              </div>
              <div class="formGrid">
                <label>Pretrain samples
                  <input data-key="pretrainSamples" type="number" min="1" step="1">
                </label>
                <label>Pretrain epochs
                  <input data-key="pretrainEpochs" type="number" min="1" step="1">
                </label>
              </div>
            </fieldset>
            <fieldset>
              <legend>Checkpoints</legend>
              <div class="formGrid">
                <label class="wide">Directory
                  <input data-key="checkpointDir" type="text">
                </label>
                <label>Every steps
                  <input data-key="checkpointEverySteps" type="number" min="0" step="1">
                </label>
                <label>Keep latest
                  <input data-key="keepCheckpoints" type="number" min="1" step="1">
                </label>
              </div>
            </fieldset>
            <fieldset>
              <legend>Telemetry</legend>
              <div class="formGrid">
                <label>NT port
                  <input data-key="advantagePort" type="number" min="1" step="1">
                </label>
                <label>Preview every
                  <input data-key="vizEverySteps" type="number" min="1" step="1">
                </label>
                <label>Preview steps
                  <input data-key="vizPreviewSteps" type="number" min="1" step="1">
                </label>
                <label>Metrics every
                  <input data-key="metricsEverySteps" type="number" min="1" step="1">
                </label>
                <label class="wide">Metrics file
                  <input data-key="metricsOut" type="text">
                </label>
              </div>
            </fieldset>
          </form>
        </section>
        <div class="charts">
          <section class="panel">
            <div class="panelHeader">
              <h2>Metrics</h2>
              <span id="metricCount">0 points</span>
            </div>
            <div class="kpis">
              <div class="kpi"><span>Steps</span><strong id="kpiSteps">0</strong></div>
              <div class="kpi"><span>Loss</span><strong id="kpiLoss">-</strong></div>
              <div class="kpi"><span>Reward</span><strong id="kpiReward">-</strong></div>
              <div class="kpi"><span>FPS</span><strong id="kpiFps">-</strong></div>
            </div>
            <div class="canvasWrap">
              <div id="chartScroll" class="chartScroll">
                <canvas id="lossCanvas"></canvas>
              </div>
            </div>
          </section>
          <div class="lower">
            <section class="panel">
              <div class="panelHeader">
                <h2>Logs</h2>
                <span id="logCount">0 lines</span>
              </div>
              <pre id="logOutput"></pre>
            </section>
            <section class="panel">
              <div class="panelHeader">
                <h2>Artifacts</h2>
                <span id="artifactCount">0 files</span>
              </div>
              <div id="artifacts" class="artifacts"></div>
            </section>
          </div>
        </div>
      </div>
    </main>
  </div>
  <script>
    const numberKeys = new Set([
      "timesteps", "nEnvs", "nSteps", "batchSize", "learningRate",
      "checkpointEverySteps", "keepCheckpoints", "pretrainSamples",
      "pretrainEpochs", "advantagePort", "vizEverySteps",
      "vizPreviewSteps", "metricsEverySteps"
    ]);
    const checkboxKeys = new Set(["heuristicPretrain", "advantageScope", "variedDefense"]);
    let defaults = {};
    let lastLogLength = 0;

    const form = document.getElementById("settingsForm");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const resetButton = document.getElementById("resetButton");
    const errorBox = document.getElementById("errorBox");
    let computeStatus = null;
    let lastMetricCount = 0;

    function setError(message) {
      errorBox.textContent = message || "";
      errorBox.style.display = message ? "block" : "none";
    }

    function fillForm(config) {
      form.querySelectorAll("[data-key]").forEach((input) => {
        const key = input.dataset.key;
        if (input.type === "checkbox") {
          input.checked = Boolean(config[key]);
        } else {
          input.value = config[key] ?? "";
        }
      });
    }

    function readForm() {
      const config = {};
      form.querySelectorAll("[data-key]").forEach((input) => {
        const key = input.dataset.key;
        if (checkboxKeys.has(key)) {
          config[key] = input.checked;
        } else if (numberKeys.has(key)) {
          config[key] = Number(input.value);
        } else {
          config[key] = input.value;
        }
      });
      return config;
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        cache: "no-store",
        ...options
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    async function loadDefaults() {
      const data = await api("/api/defaults");
      defaults = data.config;
      computeStatus = data.compute;
      fillForm(defaults);
      renderCompute(computeStatus);
    }

    async function startTraining() {
      setError("");
      try {
        await api("/api/start", {
          method: "POST",
          body: JSON.stringify(readForm())
        });
        await refresh();
      } catch (error) {
        setError(error.message);
      }
    }

    async function stopTraining() {
      setError("");
      try {
        await api("/api/stop", {method: "POST", body: "{}"});
        await refresh();
      } catch (error) {
        setError(error.message);
      }
    }

    async function refresh() {
      try {
        const status = await api("/api/status");
        renderStatus(status);
      } catch (error) {
        setError(error.message);
      }
    }

    function renderStatus(status) {
      const pill = document.getElementById("statusPill");
      pill.textContent = status.running ? "Running" : "Idle";
      pill.className = "pill " + (status.running ? "running" : "stopped");
      startButton.disabled = status.running;
      stopButton.disabled = !status.running;
      document.getElementById("pidValue").textContent = status.pid || "-";
      document.getElementById("runtimeValue").textContent = formatDuration(status.uptimeS || 0);
      document.getElementById("metricsPath").textContent = "metrics: " + (status.metricsPath || "-");
      document.getElementById("commandLine").textContent =
        "command: " + (status.command && status.command.length ? status.command.join(" ") : "-");
      if (status.compute) {
        computeStatus = status.compute;
        renderCompute(computeStatus);
      }

      const logs = status.logs || [];
      const logOutput = document.getElementById("logOutput");
      logOutput.textContent = logs.join("\n");
      document.getElementById("logCount").textContent = logs.length + " lines";
      if (logs.length !== lastLogLength) {
        logOutput.scrollTop = logOutput.scrollHeight;
        lastLogLength = logs.length;
      }

      const metrics = status.metrics || [];
      document.getElementById("metricCount").textContent = metrics.length + " points";
      renderKpis(metrics);
      drawMetricChart(
        document.getElementById("lossCanvas"),
        metrics,
        document.getElementById("chartScroll")
      );
      lastMetricCount = metrics.length;
      renderArtifacts(status.artifacts || []);
    }

    function renderCompute(compute) {
      const value = document.getElementById("computeValue");
      const cudaOption = form.querySelector('select[data-key="device"] option[value="cuda"]');
      if (!compute || !compute.torchInstalled) {
        value.textContent = "No torch";
        if (cudaOption) cudaOption.disabled = false;
        return;
      }
      if (compute.cudaAvailable) {
        value.textContent = "CUDA";
        value.title = compute.torchVersion + " " + compute.cudaDeviceName;
        if (cudaOption) cudaOption.disabled = false;
      } else {
        value.textContent = "CPU";
        value.title = compute.torchVersion + " cannot see CUDA";
        if (cudaOption) cudaOption.disabled = false;
      }
    }

    function renderKpis(metrics) {
      const latest = metrics.length ? metrics[metrics.length - 1] : {};
      document.getElementById("kpiSteps").textContent = formatNumber(latest.num_timesteps || 0);
      document.getElementById("kpiLoss").textContent = formatMetric(firstMetric(latest, ["train/loss", "train/value_loss"]));
      document.getElementById("kpiReward").textContent = formatMetric(firstMetric(latest, ["rollout/ep_rew_mean"]));
      document.getElementById("kpiFps").textContent = formatMetric(firstMetric(latest, ["time/fps"]));
    }

    function renderArtifacts(artifacts) {
      const root = document.getElementById("artifacts");
      root.innerHTML = "";
      document.getElementById("artifactCount").textContent = artifacts.length + " files";
      if (!artifacts.length) {
        const empty = document.createElement("div");
        empty.className = "artifact";
        empty.textContent = "No model files yet";
        root.appendChild(empty);
        return;
      }
      artifacts.forEach((item) => {
        const div = document.createElement("div");
        div.className = "artifact";
        const name = document.createElement("strong");
        name.textContent = item.path;
        const meta = document.createElement("span");
        meta.textContent = formatBytes(item.sizeBytes) + "  " + new Date(item.modifiedAt * 1000).toLocaleString();
        div.appendChild(name);
        div.appendChild(meta);
        root.appendChild(div);
      });
    }

    function drawMetricChart(canvas, metrics, scroller) {
      const wasPinnedRight = scroller
        ? scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 8
        : true;
      const cssWidth = chartWidthFor(metrics.length);
      const cssHeight = 260;
      const ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + "px";
      canvas.style.height = cssHeight + "px";
      const pixelWidth = Math.floor(cssWidth * ratio);
      const pixelHeight = Math.floor(cssHeight * ratio);
      if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
      if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const width = cssWidth;
      const height = cssHeight;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      const series = [
        {key: "train/loss", label: "loss", color: "#2f7d5c"},
        {key: "train/value_loss", label: "value", color: "#7456a6"},
        {key: "train/policy_gradient_loss", label: "policy", color: "#b77b1d"},
        {key: "rollout/ep_rew_mean", label: "reward", color: "#3c6e91"}
      ].map((spec) => ({
        ...spec,
        points: metrics
          .filter((row) => Number.isFinite(Number(row.num_timesteps)) && Number.isFinite(Number(row[spec.key])))
          .map((row) => ({x: Number(row.num_timesteps), y: Number(row[spec.key])}))
      })).filter((item) => item.points.length > 1);

      drawGrid(ctx, width, height);
      if (!series.length) {
        ctx.fillStyle = "#66736d";
        ctx.font = "16px Segoe UI, sans-serif";
        ctx.fillText("Waiting for trainer metrics", 28, 44);
        return;
      }

      const allPoints = series.flatMap((item) => item.points);
      const minX = Math.min(...allPoints.map((point) => point.x));
      const maxX = Math.max(...allPoints.map((point) => point.x));
      const minY = Math.min(...allPoints.map((point) => point.y));
      const maxY = Math.max(...allPoints.map((point) => point.y));
      const pad = {left: 58, right: 18, top: 22, bottom: 38};
      const ySpan = Math.max(1e-9, maxY - minY);
      const xSpan = Math.max(1, maxX - minX);
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;

      ctx.strokeStyle = "#d9e1dc";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.left, pad.top, plotW, plotH);
      ctx.fillStyle = "#66736d";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText(formatMetric(maxY), 10, pad.top + 4);
      ctx.fillText(formatMetric(minY), 10, pad.top + plotH);
      ctx.fillText(formatNumber(minX), pad.left, height - 12);
      ctx.fillText(formatNumber(maxX), width - 78, height - 12);

      series.forEach((item) => {
        ctx.beginPath();
        decimatePoints(item.points, width - pad.left - pad.right).forEach((point, index) => {
          const x = pad.left + ((point.x - minX) / xSpan) * plotW;
          const y = pad.top + plotH - ((point.y - minY) / ySpan) * plotH;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      let legendX = pad.left;
      series.forEach((item) => {
        ctx.fillStyle = item.color;
        ctx.fillRect(legendX, 8, 10, 10);
        ctx.fillStyle = "#1d2623";
        ctx.fillText(item.label, legendX + 15, 17);
        legendX += 80;
      });

      if (scroller && (wasPinnedRight || lastMetricCount === 0)) {
        scroller.scrollLeft = scroller.scrollWidth;
      }
    }

    function chartWidthFor(pointCount) {
      const visibleWidth = document.getElementById("chartScroll")?.clientWidth || 900;
      const count = Math.max(1, Number(pointCount || 0));
      let pxPerPoint = 14;
      if (count > 80) pxPerPoint = 9;
      if (count > 250) pxPerPoint = 5;
      if (count > 900) pxPerPoint = 2.5;
      if (count > 2500) pxPerPoint = 1.4;
      return Math.min(16000, Math.max(visibleWidth, 860, Math.ceil(count * pxPerPoint)));
    }

    function decimatePoints(points, plotWidth) {
      const maxPoints = Math.max(250, Math.floor(plotWidth * 1.5));
      if (points.length <= maxPoints) return points;
      const stride = Math.ceil(points.length / maxPoints);
      const sampled = [];
      for (let index = 0; index < points.length; index += stride) {
        sampled.push(points[index]);
      }
      const last = points[points.length - 1];
      if (sampled[sampled.length - 1] !== last) sampled.push(last);
      return sampled;
    }

    function drawGrid(ctx, width, height) {
      ctx.strokeStyle = "#edf2ef";
      ctx.lineWidth = 1;
      for (let x = 0; x <= width; x += 60) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y <= height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    function firstMetric(row, keys) {
      for (const key of keys) {
        if (Number.isFinite(Number(row[key]))) return Number(row[key]);
      }
      return null;
    }

    function formatMetric(value) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
      const number = Number(value);
      if (Math.abs(number) >= 1000) return number.toFixed(0);
      if (Math.abs(number) >= 10) return number.toFixed(2);
      return number.toFixed(4);
    }

    function formatNumber(value) {
      return Number(value || 0).toLocaleString();
    }

    function formatDuration(seconds) {
      const total = Math.max(0, Math.floor(seconds));
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      if (mins < 60) return mins + "m " + secs + "s";
      const hours = Math.floor(mins / 60);
      return hours + "h " + (mins % 60) + "m";
    }

    function formatBytes(bytes) {
      const value = Number(bytes || 0);
      if (value > 1024 * 1024) return (value / (1024 * 1024)).toFixed(1) + " MB";
      if (value > 1024) return (value / 1024).toFixed(1) + " KB";
      return value + " B";
    }

    startButton.addEventListener("click", startTraining);
    stopButton.addEventListener("click", stopTraining);
    resetButton.addEventListener("click", () => fillForm(defaults));
    loadDefaults().then(refresh);
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

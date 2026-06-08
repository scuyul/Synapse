from __future__ import annotations

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import shutil
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

STUDIO_DIR = REPO_ROOT / "logs" / "studio"
SAVED_CONFIGS_PATH = STUDIO_DIR / "configs.json"
RUNS_PATH = STUDIO_DIR / "runs.json"
BEST_MODEL_PATH = STUDIO_DIR / "best_model.json"

RUN_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "timesteps": 256,
        "modelOut": "models/studio_smoke",
        "nEnvs": 2,
        "nSteps": 64,
        "batchSize": 128,
        "checkpointEverySteps": 0,
        "heuristicPretrain": False,
        "advantageScope": False,
        "metricsOut": "logs/studio/smoke_metrics.jsonl",
        "metricsEverySteps": 64,
    },
    "full": {},
    "resume": {
        "resumeFrom": "models/reefscape_ppo_interrupted.zip",
        "heuristicPretrain": False,
        "modelOut": "models/reefscape_ppo_resumed",
        "metricsOut": "logs/studio/resume_metrics.jsonl",
    },
    "evaluation": {
        "timesteps": 1,
        "modelOut": "models/evaluation_only",
        "nEnvs": 1,
        "nSteps": 2,
        "batchSize": 2,
        "checkpointEverySteps": 0,
        "heuristicPretrain": False,
        "advantageScope": False,
        "metricsEverySteps": 1,
    },
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


def preset_config(name: str) -> dict[str, Any]:
    if name not in RUN_PRESETS:
        raise ValueError(f"unknown preset: {name}")
    config = dict(DEFAULT_CONFIG)
    config.update(RUN_PRESETS[name])
    if name == "evaluation":
        config["metricsOut"] = _default_metrics_path(prefix="evaluation")
    return normalize_config(config)


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
        "torchCudaRuntime": str(torch.version.cuda),
        "cudaAvailable": cuda_available,
        "cudaDeviceCount": device_count,
        "cudaDeviceName": device_name,
        "error": "",
    }


def get_gpu_stats() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 8:
        return {"available": False, "error": "unexpected nvidia-smi output"}
    return {
        "available": True,
        "name": parts[0],
        "utilizationGpuPct": _safe_float(parts[1]),
        "memoryUsedMiB": _safe_float(parts[2]),
        "memoryTotalMiB": _safe_float(parts[3]),
        "temperatureC": _safe_float(parts[4]),
        "powerDrawW": _safe_float(parts[5]),
        "powerLimitW": _safe_float(parts[6]),
        "driverVersion": parts[7],
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
        self._run_id: str | None = None

    def start_training(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        config = normalize_config(payload)
        validate_launch_config(config)
        cmd = build_train_command(config)
        metrics_path = _resolve_repo_path(config["metricsOut"])
        run_id = time.strftime("%Y%m%d_%H%M%S")
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
            self._run_id = run_id
            _record_run(
                {
                    "id": run_id,
                    "kind": "training",
                    "startedAt": self._started_at,
                    "finishedAt": None,
                    "returnCode": None,
                    "config": config,
                    "command": cmd,
                    "metricsPath": _repo_relative(metrics_path),
                    "modelOut": config["modelOut"],
                }
            )
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

    def start_artifact_process(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        path = _safe_artifact_path(payload.get("path", ""))
        cmd = _build_artifact_process(action, path)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another process is already running")
            self._logs.clear()
            self._logs.append("$ " + " ".join(cmd))
            self._command = cmd
            self._started_at = time.time()
            self._finished_at = None
            self._return_code = None
            self._last_config = dict(DEFAULT_CONFIG)
            self._metrics_path = None
            self._run_id = None
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
        threading.Thread(target=self._read_output, args=(process,), daemon=True).start()
        threading.Thread(target=self._monitor_process, args=(process,), daemon=True).start()
        return self.status()

    def stop_training(self, *, mode: str = "graceful") -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            candidate = self._process
            if candidate is not None and candidate.poll() is None:
                process = candidate
                self._logs.append(f"Stopping training ({mode})...")

        if process is None:
            return self.status()
        if mode == "kill":
            try:
                process.kill()
            except OSError:
                pass
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
        payload["gpu"] = get_gpu_stats()
        payload["metrics"] = _read_metrics(metrics_path)
        payload["artifacts"] = _list_artifacts()
        payload["runs"] = _list_runs()
        payload["savedConfigs"] = _list_saved_configs()
        payload["presets"] = sorted(RUN_PRESETS)
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
                if self._run_id is not None:
                    _finish_run(self._run_id, self._finished_at, int(return_code))


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
            self._send_json(
                {
                    "config": DEFAULT_CONFIG,
                    "compute": get_compute_status(),
                    "gpu": get_gpu_stats(),
                    "presets": sorted(RUN_PRESETS),
                    "savedConfigs": _list_saved_configs(),
                    "runs": _list_runs(),
                }
            )
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
                self._send_json(
                    self.server.state.stop_training(
                        mode=str(payload.get("mode", "graceful"))
                    )
                )
            elif path == "/api/preset":
                self._send_json({"config": preset_config(str(payload.get("name", "")))})
            elif path == "/api/config/save":
                config = normalize_config(payload.get("config", {}))
                self._send_json(
                    {
                        "savedConfigs": _save_config(
                            str(payload.get("name", "")),
                            config,
                        )
                    }
                )
            elif path == "/api/config/delete":
                self._send_json(
                    {"savedConfigs": _delete_config(str(payload.get("name", "")))}
                )
            elif path == "/api/run/duplicate":
                self._send_json(
                    {"config": _config_from_run(str(payload.get("id", "")))}
                )
            elif path == "/api/compare":
                run_ids = payload.get("runIds", [])
                if not isinstance(run_ids, list):
                    raise ValueError("runIds must be a list")
                self._send_json({"runs": _read_comparison([str(item) for item in run_ids])})
            elif path == "/api/artifact/process":
                self._send_json(self.server.state.start_artifact_process(payload))
            elif path == "/api/artifact/action":
                self._send_json(_artifact_action(payload))
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


def _default_metrics_path(*, prefix: str = "training_studio_metrics") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"logs/studio/{prefix}_{stamp}.jsonl"


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


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _list_saved_configs() -> list[dict[str, Any]]:
    configs = _load_json(SAVED_CONFIGS_PATH, [])
    if not isinstance(configs, list):
        return []
    return [
        item
        for item in configs
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("config"), dict)
    ]


def _save_config(name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("config name is required")
    configs = [item for item in _list_saved_configs() if item.get("name") != clean_name]
    configs.insert(
        0,
        {
            "name": clean_name,
            "savedAt": time.time(),
            "config": normalize_config(config),
        },
    )
    _write_json(SAVED_CONFIGS_PATH, configs[:50])
    return configs[:50]


def _delete_config(name: str) -> list[dict[str, Any]]:
    configs = [item for item in _list_saved_configs() if item.get("name") != name]
    _write_json(SAVED_CONFIGS_PATH, configs)
    return configs


def _list_runs() -> list[dict[str, Any]]:
    runs = _load_json(RUNS_PATH, [])
    return runs if isinstance(runs, list) else []


def _record_run(run: dict[str, Any]) -> None:
    runs = [item for item in _list_runs() if item.get("id") != run["id"]]
    runs.insert(0, run)
    _write_json(RUNS_PATH, runs[:100])


def _finish_run(run_id: str, finished_at: float, return_code: int) -> None:
    runs = _list_runs()
    for run in runs:
        if run.get("id") == run_id:
            run["finishedAt"] = finished_at
            run["returnCode"] = return_code
            break
    _write_json(RUNS_PATH, runs[:100])


def _config_from_run(run_id: str) -> dict[str, Any]:
    for run in _list_runs():
        if run.get("id") == run_id:
            config = dict(run.get("config", {}))
            stamp = time.strftime("%Y%m%d_%H%M%S")
            config["modelOut"] = f"{config.get('modelOut', 'models/reefscape_ppo')}_copy_{stamp}"
            config["metricsOut"] = f"logs/studio/duplicate_{stamp}.jsonl"
            return normalize_config(config)
    raise ValueError(f"run not found: {run_id}")


def _read_comparison(run_ids: list[str]) -> list[dict[str, Any]]:
    runs_by_id = {run.get("id"): run for run in _list_runs()}
    comparisons = []
    for run_id in run_ids[:4]:
        run = runs_by_id.get(run_id)
        if not run:
            continue
        metrics_path = _resolve_repo_path(run.get("metricsPath", ""))
        comparisons.append(
            {
                "id": run_id,
                "label": run.get("modelOut") or run_id,
                "metrics": _read_metrics(metrics_path, limit=2_000),
            }
        )
    return comparisons


def _list_artifacts() -> list[dict[str, Any]]:
    model_root = REPO_ROOT / "models"
    best_model = _load_json(BEST_MODEL_PATH, {})
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
                "best": best_model.get("path") == str(path.relative_to(REPO_ROOT)),
            }
        )
    artifacts.sort(key=lambda item: item["modifiedAt"], reverse=True)
    return artifacts[:20]


def _artifact_action(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action", "")).strip()
    path = _safe_artifact_path(payload.get("path", ""))
    if action == "delete":
        path.unlink()
        return {"artifacts": _list_artifacts()}
    if action == "rename":
        new_name = str(payload.get("newName", "")).strip()
        if not new_name:
            raise ValueError("newName is required")
        target = path.with_name(new_name if new_name.endswith(".zip") else f"{new_name}.zip")
        _ensure_within_models(target)
        path.rename(target)
        return {"artifacts": _list_artifacts(), "path": _repo_relative(target)}
    if action == "markBest":
        _write_json(BEST_MODEL_PATH, {"path": _repo_relative(path), "markedAt": time.time()})
        return {"artifacts": _list_artifacts()}
    raise ValueError(f"unknown artifact action: {action}")


def _build_artifact_process(action: str, path: Path) -> list[str]:
    if action == "evaluate":
        return [
            PYTHON,
            "scripts/evaluate_model.py",
            "--model",
            _repo_relative(path),
            "--episodes",
            "5",
            "--fixed-start",
            "--deterministic",
        ]
    if action == "replay":
        return [
            PYTHON,
            "scripts/run_trained_model.py",
            "--model",
            _repo_relative(path),
            "--fixed-start",
            "--loop",
            "--mental-visualizer",
        ]
    raise ValueError(f"unknown artifact process action: {action}")


def _safe_artifact_path(value: object) -> Path:
    path = _resolve_repo_path(str(value))
    _ensure_within_models(path)
    if not path.exists():
        raise ValueError(f"model not found: {_repo_relative(path)}")
    if path.suffix.lower() != ".zip":
        raise ValueError("artifact must be a .zip model")
    return path


def _ensure_within_models(path: Path) -> None:
    root = (REPO_ROOT / "models").resolve()
    resolved = path.resolve()
    if root not in (resolved, *resolved.parents):
        raise ValueError("artifact path must stay inside models/")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    .miniPanel {
      display: grid;
      gap: 6px;
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      font-size: 12px;
    }
    .progress {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #dfe8e3;
    }
    .progress > div {
      width: 0%;
      height: 100%;
      background: var(--accent);
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
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }
    .toolbar select,
    .toolbar input {
      width: auto;
      min-width: 140px;
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
    .metricGrid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 0 10px 12px;
    }
    .metricMini {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }
    .metricMini h3 {
      margin: 0;
      padding: 7px 9px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }
    .metricMini canvas {
      width: 100%;
      height: 130px;
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
    .artifactActions {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 4px;
    }
    .artifactActions button {
      padding: 4px 7px;
      font-size: 11px;
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
        <div class="progress"><div id="progressFill"></div></div>
        <div class="statusRow">
          <span>Progress</span>
          <strong id="progressValue">0%</strong>
        </div>
        <div class="miniPanel">
          <strong>GPU</strong>
          <span id="gpuValue">-</span>
          <span id="cudaHealth">CUDA health pending</span>
        </div>
        <div class="sideActions">
          <button id="startButton" class="primary" type="button">Start</button>
          <select id="stopMode">
            <option value="graceful">graceful stop</option>
            <option value="kill">hard kill</option>
          </select>
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
          <div class="toolbar">
            <select id="presetSelect">
              <option value="smoke">smoke test</option>
              <option value="full">full train</option>
              <option value="resume">resume train</option>
              <option value="evaluation">evaluation-only</option>
            </select>
            <button id="applyPresetButton" type="button">Apply preset</button>
            <input id="configName" type="text" placeholder="config name">
            <button id="saveConfigButton" type="button">Save config</button>
            <select id="savedConfigSelect"></select>
            <button id="loadConfigButton" type="button">Load</button>
            <button id="deleteConfigButton" type="button">Delete</button>
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
            <div class="toolbar">
              <select id="metricSelect"></select>
              <label>Zoom
                <input id="zoomInput" type="range" min="0.5" max="4" step="0.25" value="1">
              </label>
              <label>Smoothing
                <input id="smoothInput" type="range" min="1" max="25" step="1" value="1">
              </label>
              <button id="resetGraphButton" type="button">Reset view</button>
              <select id="compareRunSelect" multiple></select>
              <button id="duplicateRunButton" type="button">Duplicate run</button>
              <button id="compareButton" type="button">Compare</button>
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
            <div id="metricGrid" class="metricGrid"></div>
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
    const metricDefs = [
      {key: "train/loss", label: "Loss", color: "#2f7d5c"},
      {key: "rollout/ep_rew_mean", label: "Reward", color: "#3c6e91"},
      {key: "train/value_loss", label: "Value loss", color: "#7456a6"},
      {key: "train/approx_kl", label: "KL", color: "#b77b1d"},
      {key: "train/entropy_loss", label: "Entropy", color: "#9b4d57"},
      {key: "time/fps", label: "FPS", color: "#4f7f87"},
      {key: "sim/latest_other_robot_hits", label: "Collisions", color: "#b5473f"},
      {key: "sim/latest_scored_coral", label: "Scored coral", color: "#476f3f"}
    ];
    let defaults = {};
    let lastLogLength = 0;
    let comparisonRuns = [];

    const form = document.getElementById("settingsForm");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const resetButton = document.getElementById("resetButton");
    const stopMode = document.getElementById("stopMode");
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
      renderGpu(data.gpu);
      renderSavedConfigs(data.savedConfigs || []);
      renderRuns(data.runs || []);
      initializeMetricControls();
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
        await api("/api/stop", {
          method: "POST",
          body: JSON.stringify({mode: stopMode.value})
        });
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
      renderGpu(status.gpu);
      renderProgress(status);

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
      renderMetricGrid(metrics);
      renderArtifacts(status.artifacts || []);
      renderSavedConfigs(status.savedConfigs || []);
      renderRuns(status.runs || []);
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

    function renderGpu(gpu) {
      const gpuValue = document.getElementById("gpuValue");
      const health = document.getElementById("cudaHealth");
      if (!gpu || !gpu.available) {
        gpuValue.textContent = "GPU stats unavailable";
      } else {
        gpuValue.textContent =
          `${gpu.name}: ${gpu.utilizationGpuPct ?? 0}% GPU, ` +
          `${gpu.memoryUsedMiB ?? 0}/${gpu.memoryTotalMiB ?? 0} MiB, ` +
          `${gpu.temperatureC ?? 0}C, ${gpu.powerDrawW ?? 0}/${gpu.powerLimitW ?? 0} W`;
      }
      if (computeStatus && computeStatus.cudaAvailable) {
        health.textContent =
          `CUDA OK: torch ${computeStatus.torchVersion}, runtime ` +
          `${computeStatus.torchCudaRuntime}, ${computeStatus.cudaDeviceName}`;
      } else if (computeStatus && computeStatus.torchInstalled) {
        health.textContent = `CUDA unavailable to torch ${computeStatus.torchVersion}`;
      } else {
        health.textContent = "PyTorch is not installed";
      }
    }

    function renderProgress(status) {
      const total = Number(status.config?.timesteps || 0);
      const latest = (status.metrics || []).length ? status.metrics[status.metrics.length - 1] : {};
      const steps = Number(latest.num_timesteps || 0);
      const pct = total > 0 ? Math.max(0, Math.min(100, (steps / total) * 100)) : 0;
      document.getElementById("progressFill").style.width = pct.toFixed(1) + "%";
      document.getElementById("progressValue").textContent =
        total > 0 ? `${pct.toFixed(1)}%` : "until stopped";
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
        meta.textContent = (item.best ? "BEST  " : "") + formatBytes(item.sizeBytes) + "  " + new Date(item.modifiedAt * 1000).toLocaleString();
        const actions = document.createElement("div");
        actions.className = "artifactActions";
        [
          ["Evaluate", () => startArtifactProcess("evaluate", item.path)],
          ["Replay", () => startArtifactProcess("replay", item.path)],
          ["Rename", () => renameArtifact(item.path)],
          ["Delete", () => artifactAction("delete", item.path)],
          ["Best", () => artifactAction("markBest", item.path)]
        ].forEach(([label, handler]) => {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = label;
          button.addEventListener("click", handler);
          actions.appendChild(button);
        });
        div.appendChild(name);
        div.appendChild(meta);
        div.appendChild(actions);
        root.appendChild(div);
      });
    }

    async function startArtifactProcess(action, path) {
      setError("");
      try {
        await api("/api/artifact/process", {
          method: "POST",
          body: JSON.stringify({action, path})
        });
        await refresh();
      } catch (error) {
        setError(error.message);
      }
    }

    async function artifactAction(action, path, extra = {}) {
      setError("");
      try {
        await api("/api/artifact/action", {
          method: "POST",
          body: JSON.stringify({action, path, ...extra})
        });
        await refresh();
      } catch (error) {
        setError(error.message);
      }
    }

    async function renameArtifact(path) {
      const current = path.split(/[\\/]/).pop()?.replace(/\.zip$/, "") || "model";
      const newName = window.prompt("New model file name", current);
      if (!newName) return;
      await artifactAction("rename", path, {newName});
    }

    function initializeMetricControls() {
      const metricSelect = document.getElementById("metricSelect");
      metricSelect.innerHTML = "";
      metricDefs.forEach((metric) => {
        const option = document.createElement("option");
        option.value = metric.key;
        option.textContent = metric.label;
        metricSelect.appendChild(option);
      });
    }

    function renderSavedConfigs(configs) {
      const select = document.getElementById("savedConfigSelect");
      const current = select.value;
      select.innerHTML = "";
      configs.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        option._config = item.config;
        select.appendChild(option);
      });
      select.value = current;
    }

    function renderRuns(runs) {
      const select = document.getElementById("compareRunSelect");
      const selected = new Set([...select.selectedOptions].map((option) => option.value));
      select.innerHTML = "";
      runs.forEach((run) => {
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = `${run.id} ${run.modelOut || ""}`;
        option.selected = selected.has(run.id);
        select.appendChild(option);
      });
    }

    async function applyPreset() {
      const name = document.getElementById("presetSelect").value;
      const data = await api("/api/preset", {
        method: "POST",
        body: JSON.stringify({name})
      });
      fillForm(data.config);
    }

    async function saveCurrentConfig() {
      const name = document.getElementById("configName").value;
      const data = await api("/api/config/save", {
        method: "POST",
        body: JSON.stringify({name, config: readForm()})
      });
      renderSavedConfigs(data.savedConfigs || []);
    }

    async function deleteCurrentConfig() {
      const name = document.getElementById("savedConfigSelect").value;
      if (!name) return;
      const data = await api("/api/config/delete", {
        method: "POST",
        body: JSON.stringify({name})
      });
      renderSavedConfigs(data.savedConfigs || []);
    }

    function loadCurrentConfig() {
      const select = document.getElementById("savedConfigSelect");
      const selected = select.selectedOptions[0];
      if (selected && selected._config) fillForm(selected._config);
    }

    async function compareSelectedRuns() {
      const runIds = [...document.getElementById("compareRunSelect").selectedOptions]
        .map((option) => option.value);
      const data = await api("/api/compare", {
        method: "POST",
        body: JSON.stringify({runIds})
      });
      comparisonRuns = data.runs || [];
      await refresh();
    }

    async function duplicateSelectedRun() {
      const selected = document.getElementById("compareRunSelect").selectedOptions[0];
      if (!selected) return;
      const data = await api("/api/run/duplicate", {
        method: "POST",
        body: JSON.stringify({id: selected.value})
      });
      fillForm(data.config);
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

      const selectedKey = document.getElementById("metricSelect").value || "train/loss";
      const selectedMetric = metricDefs.find((metric) => metric.key === selectedKey) || metricDefs[0];
      const smoothWindow = Number(document.getElementById("smoothInput").value || 1);
      const series = [
        {
          key: selectedMetric.key,
          label: selectedMetric.label,
          color: selectedMetric.color,
          metrics,
        },
        ...comparisonRuns.map((run, index) => ({
          key: selectedMetric.key,
          label: run.label || run.id,
          color: ["#7456a6", "#b77b1d", "#b5473f", "#4f7f87"][index % 4],
          metrics: run.metrics || [],
        }))
      ].map((spec) => ({
        ...spec,
        points: smoothPoints(spec.metrics
          .filter((row) => Number.isFinite(Number(row.num_timesteps)) && Number.isFinite(Number(row[spec.key])))
          .map((row) => ({x: Number(row.num_timesteps), y: Number(row[spec.key])})), smoothWindow)
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

    function renderMetricGrid(metrics) {
      const root = document.getElementById("metricGrid");
      if (!root.dataset.ready) {
        root.innerHTML = "";
        metricDefs.forEach((metric) => {
          const card = document.createElement("div");
          card.className = "metricMini";
          const title = document.createElement("h3");
          title.textContent = metric.label;
          const canvas = document.createElement("canvas");
          canvas.dataset.metric = metric.key;
          card.appendChild(title);
          card.appendChild(canvas);
          root.appendChild(card);
        });
        root.dataset.ready = "true";
      }
      root.querySelectorAll("canvas").forEach((canvas) => {
        const metric = metricDefs.find((item) => item.key === canvas.dataset.metric);
        if (metric) drawMiniMetric(canvas, metrics, metric);
      });
    }

    function drawMiniMetric(canvas, metrics, metric) {
      const rect = canvas.getBoundingClientRect();
      const cssWidth = Math.max(220, Math.floor(rect.width || 360));
      const cssHeight = 130;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(cssWidth * ratio);
      canvas.height = Math.floor(cssHeight * ratio);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, cssWidth, cssHeight);
      const points = metrics
        .filter((row) => Number.isFinite(Number(row.num_timesteps)) && Number.isFinite(Number(row[metric.key])))
        .map((row) => ({x: Number(row.num_timesteps), y: Number(row[metric.key])}));
      if (points.length < 2) {
        ctx.fillStyle = "#66736d";
        ctx.font = "12px Segoe UI, sans-serif";
        ctx.fillText("waiting", 12, 24);
        return;
      }
      const minX = Math.min(...points.map((point) => point.x));
      const maxX = Math.max(...points.map((point) => point.x));
      const minY = Math.min(...points.map((point) => point.y));
      const maxY = Math.max(...points.map((point) => point.y));
      const pad = {left: 38, right: 10, top: 10, bottom: 24};
      const plotW = cssWidth - pad.left - pad.right;
      const plotH = cssHeight - pad.top - pad.bottom;
      const xSpan = Math.max(1, maxX - minX);
      const ySpan = Math.max(1e-9, maxY - minY);
      ctx.strokeStyle = "#edf2ef";
      ctx.strokeRect(pad.left, pad.top, plotW, plotH);
      ctx.beginPath();
      decimatePoints(points, plotW).forEach((point, index) => {
        const x = pad.left + ((point.x - minX) / xSpan) * plotW;
        const y = pad.top + plotH - ((point.y - minY) / ySpan) * plotH;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = metric.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#66736d";
      ctx.font = "11px Segoe UI, sans-serif";
      ctx.fillText(formatMetric(maxY), 4, 18);
      ctx.fillText(formatMetric(minY), 4, cssHeight - 28);
    }

    function chartWidthFor(pointCount) {
      const visibleWidth = document.getElementById("chartScroll")?.clientWidth || 900;
      const count = Math.max(1, Number(pointCount || 0));
      const zoom = Number(document.getElementById("zoomInput")?.value || 1);
      let pxPerPoint = 14 * zoom;
      if (count > 80) pxPerPoint = 9;
      if (count > 250) pxPerPoint = 5;
      if (count > 900) pxPerPoint = 2.5;
      if (count > 2500) pxPerPoint = 1.4;
      return Math.min(16000, Math.max(visibleWidth, 860, Math.ceil(count * pxPerPoint)));
    }

    function smoothPoints(points, windowSize) {
      const size = Math.max(1, Math.floor(windowSize));
      if (size <= 1 || points.length <= 2) return points;
      return points.map((point, index) => {
        const start = Math.max(0, index - size + 1);
        const slice = points.slice(start, index + 1);
        const mean = slice.reduce((sum, item) => sum + item.y, 0) / slice.length;
        return {x: point.x, y: mean};
      });
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
    document.getElementById("applyPresetButton").addEventListener("click", () => applyPreset().catch((error) => setError(error.message)));
    document.getElementById("saveConfigButton").addEventListener("click", () => saveCurrentConfig().catch((error) => setError(error.message)));
    document.getElementById("loadConfigButton").addEventListener("click", loadCurrentConfig);
    document.getElementById("deleteConfigButton").addEventListener("click", () => deleteCurrentConfig().catch((error) => setError(error.message)));
    document.getElementById("compareButton").addEventListener("click", () => compareSelectedRuns().catch((error) => setError(error.message)));
    document.getElementById("duplicateRunButton").addEventListener("click", () => duplicateSelectedRun().catch((error) => setError(error.message)));
    document.getElementById("resetGraphButton").addEventListener("click", () => {
      comparisonRuns = [];
      document.getElementById("zoomInput").value = 1;
      document.getElementById("smoothInput").value = 1;
      document.getElementById("chartScroll").scrollLeft = 0;
      refresh();
    });
    document.getElementById("metricSelect").addEventListener("change", refresh);
    document.getElementById("zoomInput").addEventListener("input", refresh);
    document.getElementById("smoothInput").addEventListener("input", refresh);
    loadDefaults().then(refresh);
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

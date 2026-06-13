from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_CONFIG: dict[str, Any] = {
    "timesteps": 100_000,
    "modelOut": "models/reefscape_ppo",
    "resumeFrom": "",
    "device": "cuda",
    "robotProfile": "sim",
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
    "visualizationBackend": "advantagescope",
    "advantagePort": 5810,
    "customUiPort": 8775,
    "customUiState": "logs/reefscape_visualizer_state.json",
    "vizEverySteps": 512,
    "vizPreviewSteps": 25,
    "variedDefense": True,
    "metricsOut": "logs/desktop_training_metrics.jsonl",
    "metricsEverySteps": 512,
    "checkpointEval": True,
    "evalEpisodes": 5,
    "bestModelOut": "models/best_reefscape_ppo",
    "evalMetricsOut": "",
}

RUN_PRESETS: dict[str, dict[str, Any]] = {
    "simple": {
        "robotProfile": "2025-robot",
        "visualizationBackend": "custom-ui",
    },
    "smoke": {
        "timesteps": 256,
        "modelOut": "models/studio_smoke",
        "nEnvs": 2,
        "nSteps": 64,
        "batchSize": 128,
        "checkpointEverySteps": 0,
        "checkpointEval": False,
        "heuristicPretrain": False,
        "advantageScope": False,
        "visualizationBackend": "none",
        "metricsOut": "logs/studio/smoke_metrics.jsonl",
        "metricsEverySteps": 64,
    },
    "full": {},
    "resume": {
        "resumeFrom": "models/reefscape_ppo_interrupted.zip",
        "heuristicPretrain": False,
        "modelOut": "models/reefscape_ppo_resumed",
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
    "customUiPort",
    "vizEverySteps",
    "vizPreviewSteps",
    "metricsEverySteps",
    "evalEpisodes",
}
FLOAT_FIELDS = {"learningRate"}
BOOL_FIELDS = {
    "heuristicPretrain",
    "advantageScope",
    "variedDefense",
    "checkpointEval",
}
STRING_FIELDS = {
    "modelOut",
    "resumeFrom",
    "device",
    "robotProfile",
    "visualizationBackend",
    "checkpointDir",
    "metricsOut",
    "customUiState",
    "bestModelOut",
    "evalMetricsOut",
}
DEVICE_CHOICES = {"auto", "cuda", "cpu"}
ROBOT_PROFILE_CHOICES = {"sim", "2025-robot"}
VISUALIZATION_BACKEND_CHOICES = {"advantagescope", "custom-ui", "both", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the native REEFSCAPE training app.")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8765, help=argparse.SUPPRESS)
    parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("The browser training studio has been replaced by the native desktop app.")
    if args.no_open:
        return 0
    exe = find_desktop_app()
    if exe is None:
        print("Build the app with: python build.py")
        print("CLI fallback: python menu.py")
        return 1
    subprocess.Popen([str(exe)], cwd=REPO_ROOT)
    print(f"Opened {exe}")
    return 0


def normalize_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULT_CONFIG)
    if payload:
        data.update(payload)
    config: dict[str, Any] = {}
    for key, default in DEFAULT_CONFIG.items():
        value = data.get(key, default)
        if key in INT_FIELDS:
            config[key] = int(value)
        elif key in FLOAT_FIELDS:
            config[key] = float(value)
        elif key in BOOL_FIELDS:
            config[key] = bool(value)
        elif key in STRING_FIELDS:
            config[key] = str(value).strip()
        else:
            config[key] = value
    if config["device"] not in DEVICE_CHOICES:
        raise ValueError("device must be auto, cuda, or cpu")
    if config["robotProfile"] not in ROBOT_PROFILE_CHOICES:
        raise ValueError("robotProfile must be sim or 2025-robot")
    if config["visualizationBackend"] not in VISUALIZATION_BACKEND_CHOICES:
        raise ValueError("visualizationBackend must be advantagescope, custom-ui, both, or none")
    if not config["advantageScope"] and config["visualizationBackend"] == "advantagescope":
        config["visualizationBackend"] = "none"
    config["advantageScope"] = config["visualizationBackend"] in {"advantagescope", "both"}
    if config["timesteps"] < 0:
        raise ValueError("timesteps must be >= 0")
    if config["learningRate"] <= 0:
        raise ValueError("learningRate must be > 0")
    return config


def preset_config(name: str) -> dict[str, Any]:
    if name not in RUN_PRESETS:
        raise ValueError(f"unknown preset: {name}")
    config = dict(DEFAULT_CONFIG)
    config.update(RUN_PRESETS[name])
    return normalize_config(config)


def validate_launch_config(
    config: dict[str, Any],
    compute_status: dict[str, Any] | None = None,
) -> None:
    status = compute_status or {"cudaAvailable": True}
    if config["device"] == "cuda" and not status.get("cudaAvailable", False):
        torch_version = status.get("torchVersion") or "not installed"
        raise ValueError(
            "CUDA was selected, but this Python environment cannot see CUDA. "
            f"torch={torch_version}."
        )


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
        "--robot-profile",
        config["robotProfile"],
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
        "--visualization-backend",
        config["visualizationBackend"],
        "--advantage-port",
        str(config["advantagePort"]),
        "--custom-ui-port",
        str(config["customUiPort"]),
        "--custom-ui-state",
        config["customUiState"],
        "--viz-every-steps",
        str(config["vizEverySteps"]),
        "--viz-preview-steps",
        str(config["vizPreviewSteps"]),
        "--metrics-out",
        config["metricsOut"],
        "--metrics-every-steps",
        str(config["metricsEverySteps"]),
    ]
    if config["visualizationBackend"] != "none":
        cmd.append("--open-visualizer")
    if config["resumeFrom"]:
        cmd.extend(["--resume-from", config["resumeFrom"]])
    if not config["heuristicPretrain"]:
        cmd.append("--skip-heuristic-pretrain")
    if config["visualizationBackend"] == "none":
        cmd.append("--no-advantagescope")
    if not config["variedDefense"]:
        cmd.append("--fixed-defense")
    if config["checkpointEval"] and config["checkpointEverySteps"] > 0:
        cmd.extend(
            [
                "--eval-checkpoints",
                "--eval-episodes",
                str(config["evalEpisodes"]),
                "--best-model-out",
                config["bestModelOut"],
            ]
        )
    return cmd


def find_desktop_app() -> Path | None:
    candidates = [
        REPO_ROOT / "ReefscapeRL.exe",
        REPO_ROOT / "builds" / "ReefscapeRL" / "ReefscapeRL.exe",
        REPO_ROOT / "builds" / "reefscape-app.exe",
        REPO_ROOT / "builds" / "reefscape-app-test.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())

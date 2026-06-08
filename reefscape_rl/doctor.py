"""Environment checks for the REEFSCAPE RL training stack."""

from __future__ import annotations

import importlib
from importlib import metadata
import shutil
import subprocess
import sys


def main() -> int:
    failures = 0
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    failures += _check_module("numpy")
    failures += _check_module("gymnasium")
    failures += _check_module("stable_baselines3")
    failures += _check_module("pygame")
    failures += _check_module("ntcore")
    failures += _check_torch()
    _check_nvidia_smi()

    if failures:
        print()
        print("One or more required runtime checks failed.")
        return 1
    print()
    print("Environment checks passed.")
    return 0


def _check_module(name: str) -> int:
    if importlib.util.find_spec(name) is None:
        print(f"{name}: missing")
        return 1
    version = _package_version(name)
    print(f"{name}: {version}")
    return 0


def _package_version(name: str) -> str:
    package_names = {
        "stable_baselines3": "stable-baselines3",
        "ntcore": "pyntcore",
    }
    try:
        return metadata.version(package_names.get(name, name))
    except metadata.PackageNotFoundError:
        return "installed"


def _check_torch() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"torch: missing ({exc})")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA: unavailable to PyTorch")
        return 1
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    return 0


def _check_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi: not found")
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError as exc:
        print(f"nvidia-smi: failed ({exc})")
        return
    if result.returncode != 0:
        print(f"nvidia-smi: failed with exit code {result.returncode}")
        return
    print(f"nvidia-smi: {result.stdout.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())

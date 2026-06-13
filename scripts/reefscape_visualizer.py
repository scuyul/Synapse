from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the native REEFSCAPE desktop visualizer."
    )
    parser.add_argument("--state-file", default="logs/reefscape_visualizer_state.json")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8775, help=argparse.SUPPRESS)
    parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("The browser visualizer has been replaced by the native desktop app.")
    print(f"Visualizer state file: {args.state_file}")
    if args.no_open:
        return 0
    exe = find_desktop_app()
    if exe is None:
        print("Build the app with: python build.py")
        print("Fallback CLI preview: python scripts/run_trained_model.py --help")
        return 1
    subprocess.Popen([str(exe)], cwd=REPO_ROOT)
    print(f"Opened {exe}")
    return 0


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

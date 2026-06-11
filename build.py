from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUILDS_DIR = REPO_ROOT / "builds"
APP_NAME = "ReefscapeRL"
APP_EXE = "ReefscapeRL.exe"
PYTHON_VERSION = "3.13.13"
PYTHON_INSTALLER = f"python-{PYTHON_VERSION}-amd64.exe"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{PYTHON_INSTALLER}"

PACKAGE_DIRS = [
    ".github",
    "app",
    "docs",
    "reefscape_rl",
    "robot_code",
    "scripts",
    "tests",
]
PACKAGE_FILES = [
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "build.bat",
    "build.py",
    "manage.py",
    "menu.py",
    "networktables.json",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the REEFSCAPE RL desktop app bundle.")
    parser.add_argument("--clean", action="store_true", help="Delete builds/ before building.")
    parser.add_argument(
        "--installer",
        action="store_true",
        help="Try to build a Windows Setup.exe with Inno Setup after packaging.",
    )
    args = parser.parse_args()

    if args.clean and BUILDS_DIR.exists():
        shutil.rmtree(BUILDS_DIR)

    BUILDS_DIR.mkdir(exist_ok=True)
    app_dir = BUILDS_DIR / APP_NAME
    if app_dir.exists():
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True)

    build_go_launcher(app_dir / APP_EXE)
    copy_app_payload(app_dir)
    write_launcher_files(app_dir)

    zip_path = make_zip(app_dir)
    print(f"Portable app folder: {app_dir}")
    print(f"Portable zip:        {zip_path}")

    if args.installer:
        ensure_python_installer()
        build_inno_installer()

    return 0


def build_go_launcher(output: Path) -> None:
    go = shutil.which("go")
    if go is None:
        raise SystemExit("Go was not found on PATH. Install Go first.")

    cmd = [go, "build", "-trimpath"]
    if os.name == "nt":
        cmd.extend(["-ldflags", "-H=windowsgui"])
    cmd.extend(["-o", str(output), "."])
    run(cmd, cwd=REPO_ROOT / "app")


def copy_app_payload(app_dir: Path) -> None:
    for relative in PACKAGE_DIRS:
        source = REPO_ROOT / relative
        if source.exists():
            shutil.copytree(
                source,
                app_dir / relative,
                ignore=shutil.ignore_patterns(
                    ".ai",
                    ".git",
                    ".gradle",
                    ".mypy_cache",
                    ".pytest_cache",
                    "__pycache__",
                    ".venv",
                    ".ruff_cache",
                    "*.egg-info",
                    "*.glb",
                    "*.pyc",
                    "*.wpi",
                    "*.wpilog",
                    "bin",
                    "build",
                    "builds",
                    "dist",
                    "env",
                    "logs",
                    "models",
                    "runs",
                    "venv",
                ),
            )

    for relative in PACKAGE_FILES:
        source = REPO_ROOT / relative
        if source.exists():
            destination = app_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def write_launcher_files(app_dir: Path) -> None:
    (app_dir / "Launch Reefscape RL.bat").write_text(
        f'@echo off\r\ncd /d "%~dp0"\r\n"{APP_EXE}"\r\n',
        encoding="utf-8",
    )
    (app_dir / "Setup Python Environment.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\setup_venv.ps1 -BootstrapPython\r\n',
        encoding="utf-8",
    )


def make_zip(app_dir: Path) -> Path:
    zip_path = BUILDS_DIR / f"{APP_NAME}-portable.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in app_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(BUILDS_DIR))
    return zip_path


def ensure_python_installer() -> Path:
    prereq_dir = BUILDS_DIR / "prereqs"
    prereq_dir.mkdir(parents=True, exist_ok=True)
    installer = prereq_dir / PYTHON_INSTALLER
    if installer.exists() and installer.stat().st_size > 0:
        print(f"Using bundled Python installer: {installer}")
        return installer

    print(f"Downloading Python {PYTHON_VERSION}:")
    print(PYTHON_URL)
    with urllib.request.urlopen(PYTHON_URL) as response:
        total_text = response.headers.get("Content-Length")
        total = int(total_text) if total_text else 0
        downloaded = 0
        chunk_size = 1024 * 256
        with installer.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                print_progress(downloaded, total)
    print()
    return installer


def print_progress(done: int, total: int) -> None:
    if total <= 0:
        print(f"\rDownloaded {done / (1024 * 1024):.1f} MB", end="", flush=True)
        return
    width = 32
    ratio = min(done / total, 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {ratio * 100:5.1f}% ({done / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)",
        end="",
        flush=True,
    )


def build_inno_installer() -> None:
    iscc = find_inno_compiler()
    if iscc is None:
        print("Inno Setup was not found on PATH; skipped Setup.exe build.")
        print("Install Inno Setup, then run: build.bat --installer")
        return

    run([iscc, str(REPO_ROOT / "installer" / "reefscape-rl.iss")], cwd=REPO_ROOT)


def find_inno_compiler() -> str | None:
    path_match = shutil.which("iscc") or shutil.which("ISCC")
    if path_match:
        return path_match

    candidates = [
        Path.home() / "AppData/Local/Programs/Inno Setup 6/ISCC.exe",
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def run(cmd: list[str], *, cwd: Path) -> None:
    print()
    print("Running:")
    print(" ".join(cmd))
    print()
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        raise SystemExit(130) from None

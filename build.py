from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUILDS_DIR = REPO_ROOT / "builds"
APP_NAME = "ReefscapeRL"
APP_EXE = "ReefscapeRL.exe"

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
    copy_python_runtime(app_dir)
    write_launcher_files(app_dir)

    zip_path = make_zip(app_dir)
    print(f"Portable app folder: {app_dir}")
    print(f"Portable zip:        {zip_path}")

    if args.installer:
        build_inno_installer()

    return 0


def build_go_launcher(output: Path) -> None:
    go = shutil.which("go")
    if go is None:
        raise SystemExit("Go was not found on PATH. Install Go first.")

    wails = find_wails_command(go)
    run([wails, "build", "-clean", "-nopackage", "-o", APP_EXE], cwd=REPO_ROOT / "app")
    built = REPO_ROOT / "app" / "build" / "bin" / APP_EXE
    if not built.exists():
        raise SystemExit(f"Wails build did not produce {built}")
    shutil.copy2(built, output)


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
                    "node_modules",
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


def copy_python_runtime(app_dir: Path) -> None:
    source = Path(sys.base_prefix).resolve()
    python_exe = source / ("python.exe" if os.name == "nt" else "bin/python")
    if not python_exe.exists():
        source = Path(sys.executable).resolve().parent
        python_exe = source / ("python.exe" if os.name == "nt" else "bin/python")
    if not python_exe.exists():
        raise SystemExit(f"Could not find base Python runtime from {sys.executable}")

    destination = app_dir / ".python"
    if destination.exists():
        shutil.rmtree(destination)

    print(f"Bundling Python runtime: {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pdb",
            ".mypy_cache",
            ".pytest_cache",
            "Doc",
            "include",
            "libs",
            "share",
            "site-packages",
            "tcl",
            "test",
            "testing",
            "tests",
        ),
    )


def write_launcher_files(app_dir: Path) -> None:
    (app_dir / "Launch Reefscape RL.bat").write_text(
        f'@echo off\r\ncd /d "%~dp0"\r\n"{APP_EXE}"\r\n',
        encoding="utf-8",
    )
    (app_dir / "Setup Python Environment.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\setup_venv.ps1 -Python ".python\\python.exe" -SkipRequirements -FastAppInstall\r\n',
        encoding="utf-8",
    )
    (app_dir / "Complete First-Time Setup.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        'powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\install_app_setup.ps1 -Python ".python\\python.exe" -InstallTrainingDependencies\r\n',
        encoding="utf-8",
    )
    (app_dir / "Install Training Dependencies.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\install_training_deps.ps1\r\n",
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


def find_wails_command(go: str) -> str:
    found = shutil.which("wails") or shutil.which("wails.exe")
    if found:
        return found

    gopath_result = subprocess.run(
        [go, "env", "GOPATH"],
        check=False,
        capture_output=True,
        text=True,
    )
    if gopath_result.returncode == 0:
        suffix = "wails.exe" if os.name == "nt" else "wails"
        candidate = Path(gopath_result.stdout.strip()) / "bin" / suffix
        if candidate.exists():
            return str(candidate)

    print("Wails CLI was not found; installing github.com/wailsapp/wails/v2/cmd/wails@v2.12.0")
    run([go, "install", "github.com/wailsapp/wails/v2/cmd/wails@v2.12.0"], cwd=REPO_ROOT / "app")
    if gopath_result.returncode == 0:
        candidate = (
            Path(gopath_result.stdout.strip())
            / "bin"
            / ("wails.exe" if os.name == "nt" else "wails")
        )
        if candidate.exists():
            return str(candidate)
    raise SystemExit("Wails CLI install finished, but the executable was not found.")


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

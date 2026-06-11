from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
NEON_BLUE = "\033[38;5;45m"
HOT_ORANGE = "\033[38;5;202m"
BOLD = "\033[1m"
RESET = "\033[0m"
MENU_COLORS = (
    "\033[38;5;45m",  # cyan
    "\033[38;5;46m",  # green
    "\033[38;5;208m",  # orange
    "\033[38;5;201m",  # magenta
    "\033[38;5;39m",  # blue
    "\033[38;5;226m",  # yellow
    "\033[38;5;129m",  # purple
    "\033[38;5;196m",  # red
    "\033[38;5;118m",  # lime
    "\033[38;5;214m",  # amber
)


def main() -> int:
    color = supports_color()
    while True:
        print()
        print(accent("REEFSCAPE RL Manage", color, bold=True))
        print(accent("===================", color))
        print(menu_line("1", "Git status", "", color, 0))
        print(menu_line("2", "Run Ruff check", "", color, 1))
        print(menu_line("3", "Check Ruff formatting", "", color, 2))
        print(menu_line("4", "Apply Ruff formatting", "", color, 3))
        print(menu_line("5", "Run unit tests", "", color, 4))
        print(menu_line("6", "Compile Python files", "", color, 5))
        print(menu_line("7", "Run full local checks", "lint + format + tests + compile", color, 6))
        print(menu_line("8", "Build release artifacts", "runs checks first", color, 7))
        print(menu_line("9", "Build release artifacts quickly", "skip checks", color, 8))
        print(menu_line("10", "Create release tag", "local tag only", color, 9))
        print(menu_line("11", "Push current branch", "", color, 10))
        print(menu_line("12", "Push release tag", "publishes GitHub Release workflow", color, 11))
        print(menu_line("13", "Exit", "", color, 12))
        choice = input(accent("Select option: ", color)).strip()

        if choice == "1":
            git_status()
        elif choice == "2":
            run_command([PYTHON, "-m", "ruff", "check", "."])
        elif choice == "3":
            run_command([PYTHON, "-m", "ruff", "format", "--check", "."])
        elif choice == "4":
            run_command([PYTHON, "-m", "ruff", "format", "."])
        elif choice == "5":
            run_command([PYTHON, "-m", "unittest", "discover", "-s", "tests"])
        elif choice == "6":
            run_command([PYTHON, "-m", "compileall", "reefscape_rl", "scripts", "tests"])
        elif choice == "7":
            run_full_checks()
        elif choice == "8":
            run_powershell_script("scripts/build_release.ps1")
        elif choice == "9":
            run_powershell_script("scripts/build_release.ps1", "-SkipChecks")
        elif choice == "10":
            create_release_tag()
        elif choice == "11":
            push_current_branch()
        elif choice == "12":
            push_release_tag()
        elif choice == "13":
            return 0
        else:
            print(accent("Invalid option.", color))


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def accent(text: str, enabled: bool, *, bold: bool = False) -> str:
    if not enabled:
        return text
    prefix = f"{BOLD}{NEON_BLUE}" if bold else NEON_BLUE
    return f"{prefix}{text}{RESET}"


def menu_line(number: str, label: str, note: str, color: bool, index: int) -> str:
    if not color:
        suffix = f" ({note})" if note else ""
        return f"{number}. {label}{suffix}"
    item_color = MENU_COLORS[index % len(MENU_COLORS)]
    number_text = f"{item_color}{BOLD}{number.rjust(2)}{RESET}"
    label_text = f"{item_color}{label}{RESET}"
    note_text = f" {HOT_ORANGE}[{note}]{RESET}" if note else ""
    return f"{number_text}  {label_text}{note_text}"


def git_status() -> None:
    run_command(["git", "status", "--short", "--branch"])


def run_full_checks() -> None:
    commands = [
        [PYTHON, "-m", "ruff", "check", "."],
        [PYTHON, "-m", "ruff", "format", "--check", "."],
        [PYTHON, "-m", "unittest", "discover", "-s", "tests"],
        [PYTHON, "-m", "compileall", "reefscape_rl", "scripts", "tests"],
    ]
    for cmd in commands:
        if run_command(cmd) != 0:
            print("Stopped because a check failed.")
            return


def run_powershell_script(script_path: str, *args: str) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        print("Could not find pwsh or powershell on PATH.")
        return

    cmd = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(REPO_ROOT / script_path),
        *args,
    ]
    run_command(cmd)


def create_release_tag() -> None:
    version = get_project_version()
    default_tag = f"v{version}"
    tag = prompt_text("Release tag", default_tag)
    message = prompt_text("Tag message", f"Release {tag}")

    if not prompt_bool(f"Create local tag {tag}", False):
        print("Tag creation cancelled.")
        return

    run_command(["git", "tag", "-a", tag, "-m", message])


def push_current_branch() -> None:
    branch = command_output(["git", "branch", "--show-current"]).strip()
    if not branch:
        print("Could not determine the current branch.")
        return
    if not prompt_bool(f"Push current branch '{branch}' to origin", False):
        print("Branch push cancelled.")
        return
    run_command(["git", "push", "origin", branch])


def push_release_tag() -> None:
    version = get_project_version()
    tag = prompt_text("Release tag to push", f"v{version}")
    if not prompt_bool(f"Push tag {tag} to origin", False):
        print("Tag push cancelled.")
        return
    run_command(["git", "push", "origin", tag])


def get_project_version() -> str:
    code = "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
    return command_output([PYTHON, "-c", code]).strip()


def prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        print("Enter y or n.")


def run_command(cmd: list[str]) -> int:
    print()
    print("Running:")
    print(" ".join(cmd))
    print(flush=True)
    try:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    except FileNotFoundError:
        print(f"Could not find command: {cmd[0]}")
        return 127
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    return completed.returncode


def command_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)


if __name__ == "__main__":
    raise SystemExit(main())

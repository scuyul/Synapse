from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
NEON_ORANGE = "\033[38;5;208m"
HOT_ORANGE = "\033[38;5;202m"
PURPLE = "\033[38;5;141m"
BOLD = "\033[1m"
RESET = "\033[0m"
MENU_COLORS = (
    "\033[38;5;51m",  # cyan
    "\033[38;5;208m",  # orange
    "\033[38;5;46m",  # green
    "\033[38;5;201m",  # magenta
    "\033[38;5;39m",  # blue
    "\033[38;5;226m",  # yellow
    "\033[38;5;129m",  # purple
    "\033[38;5;196m",  # red
    "\033[38;5;118m",  # lime
    "\033[38;5;214m",  # amber
    "\033[38;5;87m",  # aqua
    "\033[38;5;165m",  # pink
)


def main() -> int:
    color = supports_color()
    while True:
        print()
        print(accent("REEFSCAPE RL Control", color, bold=True))
        print(accent("====================", color))
        print(menu_line("1", "Open Training Studio", "recommended", color, 0))
        print(menu_line("2", "Simple CUDA train", "recommended settings", color, 1))
        print(menu_line("3", "Check CUDA / RTX 4060", "", color, 2))
        print(menu_line("4", "Run tests", "", color, 3))
        print(menu_line("5", "Watch heuristic simulator in AdvantageScope", "", color, 4))
        print(menu_line("6", "Train PPO from menu fallback", "", color, 5))
        print(menu_line("7", "Resume PPO from menu fallback", "", color, 6))
        print(menu_line("8", "Watch trained model in AdvantageScope", "", color, 7))
        print(menu_line("9", "Quick train smoke test", "", color, 8))
        print(menu_line("10", "Quick trained-model smoke run", "", color, 9))
        print(menu_line("11", "Generate rollout CSV log", "", color, 10))
        print(menu_line("12", "Exit", "", color, 11))
        choice = input(accent("Select option: ", color)).strip()

        if choice == "1":
            run_training_studio()
        elif choice == "2":
            run_simple_training()
        elif choice == "3":
            check_cuda()
        elif choice == "4":
            run_command([PYTHON, "-m", "unittest", "discover", "-s", "tests"])
        elif choice == "5":
            run_live_heuristic()
        elif choice == "6":
            run_training()
        elif choice == "7":
            run_training(resume=True)
        elif choice == "8":
            run_trained_model()
        elif choice == "9":
            run_training(smoke=True)
        elif choice == "10":
            run_trained_model(smoke=True)
        elif choice == "11":
            run_rollout_log()
        elif choice == "12":
            return 0
        else:
            print(accent("Invalid option.", color))


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def accent(text: str, enabled: bool, *, bold: bool = False) -> str:
    if not enabled:
        return text
    prefix = f"{BOLD}{NEON_ORANGE}" if bold else NEON_ORANGE
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


def run_live_heuristic() -> None:
    policy = prompt_choice("Policy", "heuristic", {"heuristic", "random"})
    seed = prompt_int("Seed", 1)
    port = prompt_int("AdvantageScope NT port", 5810)
    speed = prompt_float("Playback speed", 1.0)
    loop = prompt_bool("Loop episodes", True)
    fixed_start = prompt_bool("Fixed start", True)
    xbox_defense = prompt_bool("Drive defense robot with Xbox controller", False)

    cmd = [
        PYTHON,
        "scripts/live_advantagescope.py",
        "--policy",
        policy,
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--speed",
        str(speed),
    ]
    if loop:
        cmd.append("--loop")
    if fixed_start:
        cmd.append("--fixed-start")
    if xbox_defense:
        cmd.append("--xbox-defense")
    run_command(cmd)


def run_training(*, smoke: bool = False, resume: bool = False) -> None:
    timesteps = prompt_int("Training timesteps (0 = until Ctrl+C)", 256 if smoke else 100_000)
    model_out = prompt_text(
        "Model output path", "models/smoke_train_viz" if smoke else "models/reefscape_ppo"
    )
    resume_from = ""
    if resume:
        resume_from = prompt_text(
            "Resume from model/checkpoint .zip", "models/reefscape_ppo_interrupted.zip"
        )
    device = prompt_choice("Device", "cuda", {"auto", "cuda", "cpu"})
    n_envs = prompt_int("Parallel envs", 2 if smoke else 8)
    n_steps = prompt_int("PPO rollout steps per env", 64 if smoke else 512)
    batch_size = prompt_int("PPO batch size", 128 if smoke else 1024)
    learning_rate = prompt_float("Learning rate", 0.0003)
    checkpoint_dir = prompt_text("Checkpoint directory", "models/checkpoints")
    checkpoint_every = prompt_int("Checkpoint every N steps", 10_000)
    keep_checkpoints = prompt_int("Keep latest N checkpoints", 2)
    heuristic_pretrain = prompt_bool("Pretrain from heuristic first", not resume)
    pretrain_samples = prompt_int("Heuristic pretrain samples", 50_000)
    pretrain_epochs = prompt_int("Heuristic pretrain epochs", 10)
    advantagescope = prompt_bool("Stream training preview to AdvantageScope", True)
    varied_defense = prompt_bool("Randomize defense bot during training", True)
    port = prompt_int("AdvantageScope NT port", 5810)
    viz_every = prompt_int("Preview every N training steps", 64 if smoke else 512)
    viz_steps = prompt_int("Preview sim steps per update", 5 if smoke else 25)

    cmd = [
        PYTHON,
        "scripts/train_ppo.py",
        "--timesteps",
        str(timesteps),
        "--model-out",
        model_out,
        "--device",
        device,
        "--n-envs",
        str(n_envs),
        "--n-steps",
        str(n_steps),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--checkpoint-dir",
        checkpoint_dir,
        "--checkpoint-every-steps",
        str(checkpoint_every),
        "--keep-checkpoints",
        str(keep_checkpoints),
        "--pretrain-heuristic-samples",
        str(pretrain_samples),
        "--pretrain-heuristic-epochs",
        str(pretrain_epochs),
        "--advantage-port",
        str(port),
        "--viz-every-steps",
        str(viz_every),
        "--viz-preview-steps",
        str(viz_steps),
    ]
    if resume_from:
        cmd.extend(["--resume-from", resume_from])
    if not heuristic_pretrain:
        cmd.append("--skip-heuristic-pretrain")
    if not advantagescope:
        cmd.append("--no-advantagescope")
    if not varied_defense:
        cmd.append("--fixed-defense")
    run_command(cmd)


def run_simple_training() -> None:
    cmd = [
        PYTHON,
        "scripts/train_ppo.py",
        "--timesteps",
        "100000",
        "--device",
        "cuda",
        "--n-envs",
        "8",
        "--n-steps",
        "512",
        "--batch-size",
        "1024",
        "--learning-rate",
        "0.0003",
        "--checkpoint-dir",
        "models/checkpoints",
        "--checkpoint-every-steps",
        "10000",
        "--keep-checkpoints",
        "2",
        "--eval-checkpoints",
        "--eval-episodes",
        "5",
        "--best-model-out",
        "models/best_reefscape_ppo",
    ]
    run_command(cmd)


def run_trained_model(*, smoke: bool = False) -> None:
    model = prompt_text("Model path", "models/reefscape_ppo.zip")
    episodes = prompt_int("Episodes", 1)
    seed = prompt_int("Seed", 1)
    port = prompt_int("AdvantageScope NT port", 5810)
    speed = prompt_float("Playback speed", 100.0 if smoke else 1.0)
    loop = prompt_bool("Loop episodes", not smoke)
    fixed_start = prompt_bool("Fixed start", True)
    deterministic = prompt_bool("Deterministic actions", False)
    auto_mechanisms = prompt_bool("Auto-run mechanisms", False)
    mental_visualizer = prompt_bool("Show AI mental visualizer telemetry", True)
    xbox_defense = prompt_bool("Drive defense robot with Xbox controller", False)

    cmd = [
        PYTHON,
        "scripts/run_trained_model.py",
        "--model",
        model,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--speed",
        str(speed),
    ]
    if loop:
        cmd.append("--loop")
    if fixed_start:
        cmd.append("--fixed-start")
    if deterministic:
        cmd.append("--deterministic")
    if auto_mechanisms:
        cmd.append("--auto-mechanisms")
    if mental_visualizer:
        cmd.append("--mental-visualizer")
    if xbox_defense:
        cmd.append("--xbox-defense")
    run_command(cmd)


def run_rollout_log() -> None:
    policy = prompt_choice("Policy", "heuristic", {"heuristic", "random"})
    episodes = prompt_int("Episodes", 1)
    seed = prompt_int("Seed", 1)
    duration = prompt_float("Episode duration seconds", 150.0)
    max_coral = prompt_int("Max coral scored", 12)
    out = prompt_text("Output CSV path", "logs/heuristic_rollout.csv")
    fixed_start = prompt_bool("Fixed start", True)

    cmd = [
        PYTHON,
        "scripts/run_rollout.py",
        "--policy",
        policy,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--duration",
        str(duration),
        "--max-coral",
        str(max_coral),
        "--out",
        out,
    ]
    if fixed_start:
        cmd.append("--fixed-start")
    run_command(cmd)


def run_training_studio() -> None:
    port = prompt_int("Training studio port", 8765)
    open_browser = prompt_bool("Open browser automatically", True)
    cmd = [
        PYTHON,
        "scripts/training_studio.py",
        "--port",
        str(port),
    ]
    if not open_browser:
        cmd.append("--no-open")
    run_command(cmd)


def check_cuda() -> None:
    code = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('cuda_available', torch.cuda.is_available()); "
        "print('device_count', torch.cuda.device_count()); "
        "print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
    )
    run_command([PYTHON, "-c", code])


def prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_int(label: str, default: int) -> int:
    while True:
        value = prompt_text(label, str(default))
        try:
            return int(value)
        except ValueError:
            print("Enter an integer.")


def prompt_float(label: str, default: float) -> float:
    while True:
        value = prompt_text(label, str(default))
        try:
            return float(value)
        except ValueError:
            print("Enter a number.")


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


def prompt_choice(label: str, default: str, choices: set[str]) -> str:
    choices_text = "/".join(sorted(choices))
    while True:
        value = input(f"{label} ({choices_text}) [{default}]: ").strip()
        if not value:
            return default
        if value in choices:
            return value
        print(f"Choose one of: {choices_text}")


def run_command(cmd: list[str]) -> None:
    print()
    print("Running:")
    print(" ".join(cmd))
    print()
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    raise SystemExit(main())

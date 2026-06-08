from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_UNTIL_STOPPED_TIMESTEPS = 2_147_483_647


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the REEFSCAPE environment.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Training timesteps. Use 0 to train until stopped with Ctrl+C.",
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/reefscape_ppo"))
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--checkpoint-every-steps", type=int, default=10_000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--metrics-every-steps", type=int, default=512)
    parser.add_argument("--pretrain-heuristic-samples", type=int, default=50_000)
    parser.add_argument("--pretrain-heuristic-epochs", type=int, default=10)
    parser.add_argument("--skip-heuristic-pretrain", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Compatibility flag. AdvantageScope preview is already enabled unless --no-advantagescope is set.",
    )
    parser.add_argument("--no-advantagescope", action="store_true")
    parser.add_argument("--advantage-port", type=int, default=5810)
    parser.add_argument("--viz-every-steps", type=int, default=512)
    parser.add_argument("--viz-preview-steps", type=int, default=25)
    parser.add_argument(
        "--fixed-defense",
        action="store_true",
        help="Disable randomized defense-bot path/speed/start during training.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="cuda",
        help="Training device. Defaults to CUDA; use 'auto' or 'cpu' as fallback.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.env_util import make_vec_env
        from reefscape_rl.env import ReefscapeEnvConfig
        from reefscape_rl.gymnasium_env import GymnasiumReefscapeEnv
        from reefscape_rl.training_viz import (
            AdvantageScopeTrainingCallback,
            TrainingVisualizationConfig,
        )
        from reefscape_rl.imitation import ImitationConfig, pretrain_from_heuristic
    except ImportError as exc:
        print(exc)
        print("Install dependencies with: python -m pip install -r .\\requirements.txt")
        return 2

    args = parse_args()
    try:
        total_timesteps = _resolve_total_timesteps(args.timesteps)
    except ValueError as exc:
        print(exc)
        return 2
    if args.timesteps == 0:
        print("Training until stopped. Press Ctrl+C to save an interrupted model.")

    rollout_size = args.n_envs * args.n_steps
    batch_size = min(args.batch_size, rollout_size)
    if batch_size != args.batch_size:
        print(f"Reduced batch size to rollout size: {batch_size}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but this PyTorch build cannot see CUDA.")
        print(f"torch version: {torch.__version__}")
        return 2

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    def make_env():
        return GymnasiumReefscapeEnv(
            config=None
            if not args.fixed_defense
            else ReefscapeEnvConfig(
                auto_mechanisms=False,
                randomize_other_robot_start=False,
                randomize_other_robot_behavior=False,
            )
        )

    env = make_vec_env(make_env, n_envs=args.n_envs)
    if args.resume_from is not None:
        if not args.resume_from.exists():
            print(f"Resume model not found: {args.resume_from}")
            return 2
        model = PPO.load(args.resume_from, env=env, device=device)
        print(f"Resumed model: {args.resume_from}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            n_steps=args.n_steps,
            batch_size=batch_size,
            learning_rate=args.learning_rate,
            policy_kwargs={"net_arch": [256, 256]},
        )
        if not args.skip_heuristic_pretrain:
            pretrain_from_heuristic(
                model,
                ImitationConfig(
                    samples=args.pretrain_heuristic_samples,
                    epochs=args.pretrain_heuristic_epochs,
                ),
            )

    callbacks: list[BaseCallback] = []
    if not args.no_advantagescope:
        callbacks.append(
            AdvantageScopeTrainingCallback(
                TrainingVisualizationConfig(
                    port=args.advantage_port,
                    every_steps=args.viz_every_steps,
                    preview_steps=args.viz_preview_steps,
                )
            )
        )
        print("Training visualization enabled for AdvantageScope.")
        print(f"Connect AdvantageScope to NetworkTables at 127.0.0.1:{args.advantage_port}")

    if args.checkpoint_every_steps > 0:
        callbacks.append(
            RotatingCheckpointCallback(
                checkpoint_dir=args.checkpoint_dir,
                every_steps=args.checkpoint_every_steps,
                keep=args.keep_checkpoints,
            )
        )

    if args.metrics_out is not None:
        callbacks.append(
            MetricsJsonlCallback(
                path=args.metrics_out,
                every_steps=args.metrics_every_steps,
            )
        )
        print(f"Training metrics will be written to {args.metrics_out}")

    callback = CallbackList(callbacks) if callbacks else None
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=args.resume_from is None,
        )
    except KeyboardInterrupt:
        interrupt_path = args.model_out.with_name(f"{args.model_out.name}_interrupted")
        model.save(interrupt_path)
        print()
        print(f"Training interrupted. Saved current model to {interrupt_path}.zip")
        return 130

    model.save(args.model_out)
    print(f"Wrote {args.model_out}.zip")
    return 0


def _resolve_total_timesteps(timesteps: int) -> int:
    if timesteps < 0:
        raise ValueError("--timesteps must be >= 0")
    if timesteps == 0:
        return TRAIN_UNTIL_STOPPED_TIMESTEPS
    return timesteps


class RotatingCheckpointCallback(BaseCallback):
    def __init__(self, *, checkpoint_dir: Path, every_steps: int, keep: int):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every_steps = every_steps
        self.keep = max(1, keep)
        self.saved: list[Path] = []

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.every_steps != 0:
            return True
        path = self.checkpoint_dir / f"reefscape_ppo_step_{self.num_timesteps}"
        self.model.save(path)
        zip_path = path.with_suffix(".zip")
        self.saved.append(zip_path)
        print(f"Checkpoint saved: {zip_path}")
        while len(self.saved) > self.keep:
            old = self.saved.pop(0)
            _delete_checkpoint_if_possible(old)
        return True


def _delete_checkpoint_if_possible(path: Path) -> None:
    if not path.exists():
        return
    for _ in range(3):
        try:
            path.unlink()
            print(f"Deleted old checkpoint: {path}")
            return
        except PermissionError:
            time.sleep(0.1)
    print(f"Warning: could not delete old checkpoint because it is locked: {path}")


class MetricsJsonlCallback(BaseCallback):
    def __init__(self, *, path: Path, every_steps: int):
        super().__init__()
        self.path = path
        self.every_steps = max(1, every_steps)
        self._next_step = self.every_steps
        self._start_wall_time = 0.0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._start_wall_time = time.monotonic()
        self._write(event="start")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_step:
            return True
        while self._next_step <= self.num_timesteps:
            self._next_step += self.every_steps
        self._write(event="step")
        return True

    def _on_training_end(self) -> None:
        self._write(event="end")

    def _write(self, *, event: str) -> None:
        payload: dict[str, float | int | str] = {
            "event": event,
            "num_timesteps": int(self.num_timesteps),
            "elapsed_s": max(0.0, time.monotonic() - self._start_wall_time),
        }
        for key, value in self.logger.name_to_value.items():
            number = _to_finite_float(value)
            if number is not None:
                payload[key] = number

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")


def _to_finite_float(value: object) -> float | None:
    if hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


if __name__ == "__main__":
    raise SystemExit(main())

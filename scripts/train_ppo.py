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
    parser.add_argument(
        "--eval-checkpoints",
        action="store_true",
        help="Run a deterministic evaluation after each checkpoint and update the best model.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--best-model-out", type=Path, default=Path("models/best_reefscape_ppo"))
    parser.add_argument("--eval-metrics-out", type=Path, default=None)
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
    parser.add_argument(
        "--visualization-backend",
        choices=("advantagescope", "custom-ui", "both", "none"),
        default="advantagescope",
        help="Live preview backend for training rollouts.",
    )
    parser.add_argument(
        "--open-visualizer",
        action="store_true",
        help="Open the selected visualizer app when training starts.",
    )
    parser.add_argument("--advantage-port", type=int, default=5810)
    parser.add_argument("--custom-ui-port", type=int, default=8775)
    parser.add_argument(
        "--custom-ui-state",
        type=Path,
        default=Path("logs/reefscape_visualizer_state.json"),
        help="JSON state file used by the custom REEFSCAPE visualizer.",
    )
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
    parser.add_argument(
        "--robot-profile",
        choices=("sim", "2025-robot"),
        default="sim",
        help="Robot dynamics profile. Use 2025-robot to train against copied robot constants.",
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
            CustomUiTrainingCallback,
            CustomUiTrainingConfig,
            TrainingVisualizationConfig,
        )
        from reefscape_rl.imitation import ImitationConfig, pretrain_from_heuristic
        from reefscape_rl.robot_integration import apply_profile_to_env_config, load_robot_profile
    except ImportError as exc:
        print(exc)
        print("Install dependencies with: python -m pip install -r .\\requirements.txt")
        return 2

    args = parse_args()
    try:
        robot_profile = load_robot_profile(args.robot_profile)
    except (OSError, ValueError) as exc:
        print(exc)
        return 2
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
    if robot_profile is not None:
        print(
            "Using robot profile: "
            f"{robot_profile.name} "
            f"({robot_profile.max_linear_speed_mps:.2f} m/s, "
            f"{robot_profile.max_angular_speed_radps:.2f} rad/s)"
        )

    def make_env():
        env_config = (
            None
            if not args.fixed_defense
            else ReefscapeEnvConfig(
                auto_mechanisms=False,
                randomize_other_robot_start=False,
                randomize_other_robot_behavior=False,
            )
        )
        if robot_profile is not None:
            env_config = apply_profile_to_env_config(
                env_config or ReefscapeEnvConfig(), robot_profile
            )
        return GymnasiumReefscapeEnv(
            config=env_config,
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
    visualization_backend = "none" if args.no_advantagescope else args.visualization_backend
    use_advantagescope = visualization_backend in {"advantagescope", "both"}
    use_custom_ui = visualization_backend in {"custom-ui", "both"}
    if use_advantagescope:
        callbacks.append(
            AdvantageScopeTrainingCallback(
                TrainingVisualizationConfig(
                    port=args.advantage_port,
                    every_steps=args.viz_every_steps,
                    preview_steps=args.viz_preview_steps,
                    open_app=args.open_visualizer,
                )
            )
        )
        print("Training visualization enabled for AdvantageScope.")
        print(f"Connect AdvantageScope to NetworkTables at 127.0.0.1:{args.advantage_port}")
    if use_custom_ui:
        callbacks.append(
            CustomUiTrainingCallback(
                CustomUiTrainingConfig(
                    port=args.custom_ui_port,
                    state_path=args.custom_ui_state,
                    every_steps=args.viz_every_steps,
                    preview_steps=args.viz_preview_steps,
                    open_app=args.open_visualizer,
                )
            )
        )
        print("Training visualization enabled for the native REEFSCAPE desktop visualizer.")
        print(f"Desktop visualizer state: {args.custom_ui_state}")

    if args.checkpoint_every_steps > 0:
        callbacks.append(
            RotatingCheckpointCallback(
                checkpoint_dir=args.checkpoint_dir,
                every_steps=args.checkpoint_every_steps,
                keep=args.keep_checkpoints,
                eval_after_save=args.eval_checkpoints,
                eval_episodes=args.eval_episodes,
                best_model_out=args.best_model_out,
                eval_metrics_out=args.eval_metrics_out,
                make_eval_env=make_env,
            )
        )
        if args.eval_checkpoints:
            print(
                "Checkpoint evaluation enabled; best model will be written to "
                f"{args.best_model_out}.zip"
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
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        every_steps: int,
        keep: int,
        eval_after_save: bool = False,
        eval_episodes: int = 5,
        best_model_out: Path | None = None,
        eval_metrics_out: Path | None = None,
        make_eval_env=None,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every_steps = every_steps
        self.keep = max(1, keep)
        self.saved: list[Path] = []
        self.eval_after_save = eval_after_save
        self.eval_episodes = max(1, eval_episodes)
        self.best_model_out = best_model_out
        self.eval_metrics_out = eval_metrics_out
        self.make_eval_env = make_eval_env
        self.best_score: tuple[float, float, float, float] | None = None

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.eval_metrics_out is not None:
            self.eval_metrics_out.parent.mkdir(parents=True, exist_ok=True)
            self.eval_metrics_out.write_text("", encoding="utf-8")
        if self.best_model_out is not None:
            self.best_model_out.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.every_steps != 0:
            return True
        path = self.checkpoint_dir / f"reefscape_ppo_step_{self.num_timesteps}"
        self.model.save(path)
        zip_path = path.with_suffix(".zip")
        self.saved.append(zip_path)
        print(f"Checkpoint saved: {zip_path}")
        if self.eval_after_save:
            self._evaluate_checkpoint(zip_path)
        while len(self.saved) > self.keep:
            old = self.saved.pop(0)
            _delete_checkpoint_if_possible(old)
        return True

    def _evaluate_checkpoint(self, checkpoint_path: Path) -> None:
        if self.make_eval_env is None:
            print("Warning: checkpoint evaluation skipped because no eval env factory exists.")
            return
        result = _evaluate_model(
            self.model,
            make_eval_env=self.make_eval_env,
            episodes=self.eval_episodes,
        )
        result.update(
            {
                "event": "checkpoint_eval",
                "num_timesteps": int(self.num_timesteps),
                "checkpoint": str(checkpoint_path),
            }
        )
        score = (
            float(result["mean_scored_coral"]),
            float(result["mean_return"]),
            -float(result["mean_hard_hits"]),
            -float(result["mean_hits"]),
        )
        is_best = self.best_score is None or score > self.best_score
        result["best"] = is_best
        if is_best:
            self.best_score = score
            if self.best_model_out is not None:
                self.model.save(self.best_model_out)
                metadata_path = self.best_model_out.with_suffix(".json")
                metadata_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"New best model: {self.best_model_out}.zip")
        if self.eval_metrics_out is not None:
            with self.eval_metrics_out.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            "Checkpoint eval: "
            f"return={result['mean_return']:.3f}, "
            f"scored={result['mean_scored_coral']:.2f}, "
            f"hits={result['mean_hits']:.2f}, "
            f"hard_hits={result['mean_hard_hits']:.2f}"
        )


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


def _evaluate_model(model, *, make_eval_env, episodes: int) -> dict[str, float]:
    returns: list[float] = []
    scored: list[float] = []
    hits: list[float] = []
    hard_hits: list[float] = []
    for episode in range(episodes):
        env = make_eval_env()
        observation, info = env.reset(seed=10_000 + episode)
        done = False
        episode_return = 0.0
        last_info = info if isinstance(info, dict) else {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)
            if isinstance(info, dict):
                last_info = info
        close = getattr(env, "close", None)
        if callable(close):
            close()
        returns.append(episode_return)
        scored.append(float(last_info.get("scored_coral", 0.0)))
        hits.append(float(last_info.get("other_robot_hits", 0.0)))
        hard_hits.append(float(last_info.get("other_robot_hard_hits", 0.0)))
    return {
        "mean_return": _mean(returns),
        "mean_scored_coral": _mean(scored),
        "mean_hits": _mean(hits),
        "mean_hard_hits": _mean(hard_hits),
        "best_return": max(returns) if returns else 0.0,
        "best_scored_coral": max(scored) if scored else 0.0,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class MetricsJsonlCallback(BaseCallback):
    def __init__(self, *, path: Path, every_steps: int):
        super().__init__()
        self.path = path
        self.every_steps = max(1, every_steps)
        self._next_step = self.every_steps
        self._start_wall_time = 0.0
        self._latest_episode_reward: float | None = None
        self._latest_scored_coral: int | None = None
        self._latest_target_level_code: int | None = None
        self._latest_scored_level_code: int | None = None
        self._latest_other_robot_hits: int | None = None
        self._latest_other_robot_hard_hits: int | None = None
        self._latest_frozen_time_s: float | None = None

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._start_wall_time = time.monotonic()
        self._write(event="start")

    def _on_step(self) -> bool:
        self._capture_infos()
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
        extras = {
            "sim/latest_episode_reward": self._latest_episode_reward,
            "sim/latest_scored_coral": self._latest_scored_coral,
            "sim/latest_target_level_code": self._latest_target_level_code,
            "sim/latest_scored_level_code": self._latest_scored_level_code,
            "sim/latest_other_robot_hits": self._latest_other_robot_hits,
            "sim/latest_other_robot_hard_hits": self._latest_other_robot_hard_hits,
            "sim/latest_frozen_time_s": self._latest_frozen_time_s,
        }
        for key, value in extras.items():
            if value is not None:
                payload[key] = value

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _capture_infos(self) -> None:
        infos = self.locals.get("infos", ())
        for info in infos:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if isinstance(episode, dict):
                reward = _to_finite_float(episode.get("r"))
                if reward is not None:
                    self._latest_episode_reward = reward
            if "scored_coral" in info:
                self._latest_scored_coral = int(info["scored_coral"])
            if "target_level_code" in info:
                self._latest_target_level_code = int(info["target_level_code"])
            if info.get("scored_level_code", 0):
                self._latest_scored_level_code = int(info["scored_level_code"])
            if "other_robot_hits" in info:
                self._latest_other_robot_hits = int(info["other_robot_hits"])
            if "other_robot_hard_hits" in info:
                self._latest_other_robot_hard_hits = int(info["other_robot_hard_hits"])
            frozen_time = _to_finite_float(info.get("frozen_time_s"))
            if frozen_time is not None:
                self._latest_frozen_time_s = frozen_time


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

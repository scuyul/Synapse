from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.mental_visualizer import (
    MentalVisualizerPublisher,
    build_mental_snapshot,
)
from reefscape_rl.nt_publisher import AdvantageScopeNtPublisher
from reefscape_rl.robot_integration import apply_profile_to_env_config, load_robot_profile
from reefscape_rl.visualizer_snapshot import build_visualizer_snapshot
from reefscape_rl.xbox_controller import XboxController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved PPO model and stream it to a visualizer."
    )
    parser.add_argument("--model", type=Path, default=Path("models/reefscape_ppo.zip"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=5810)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--visualization-backend",
        choices=("advantagescope", "custom-ui", "both", "none"),
        default="advantagescope",
    )
    parser.add_argument(
        "--custom-ui-state",
        type=Path,
        default=Path("logs/reefscape_visualizer_state.json"),
        help="JSON state path consumed by the native Field tab.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--auto-mechanisms", action="store_true")
    parser.add_argument("--manual-mechanisms", action="store_true")
    parser.add_argument("--raw-actions", action="store_true")
    parser.add_argument(
        "--robot-profile",
        choices=("sim", "2025-robot"),
        default="sim",
        help="Robot dynamics profile used by the replay environment.",
    )
    parser.add_argument(
        "--mental-visualizer",
        action="store_true",
        help="Publish demo-friendly /AI/Mental/* topics and /AdvantageScope/AIAttentionPose.",
    )
    parser.add_argument(
        "--xbox-defense",
        action="store_true",
        help="Use the first Xbox controller to manually drive /AdvantageScope/OtherRobotPose.",
    )
    parser.add_argument("--xbox-deadband", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        print(exc)
        print("Install dependencies with: python -m pip install -r .\\requirements.txt")
        return 2

    args = parse_args()
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return 2
    try:
        robot_profile = load_robot_profile(args.robot_profile)
    except (OSError, ValueError) as exc:
        print(exc)
        return 2

    model = PPO.load(args.model)
    env_config = ReefscapeEnvConfig(
        randomize_start=not args.fixed_start,
        auto_mechanisms=args.auto_mechanisms and not args.manual_mechanisms,
        other_robot_manual_control=args.xbox_defense,
    )
    env_config = apply_profile_to_env_config(env_config, robot_profile)
    env = ReefscapeEnv(env_config)
    action_adapter = None if args.raw_actions else ResidualHeuristicActionAdapter()
    use_advantagescope = args.visualization_backend in {"advantagescope", "both"}
    use_custom_ui = args.visualization_backend in {"custom-ui", "both"}
    publisher = (
        AdvantageScopeNtPublisher.start_server(port=args.port)
        if use_advantagescope
        else None
    )
    mental_visualizer = (
        MentalVisualizerPublisher(publisher.inst)
        if args.mental_visualizer and publisher is not None
        else None
    )
    custom_ui = CustomUiReplayWriter(args.custom_ui_state, args.model.stem) if use_custom_ui else None
    controller = None
    if args.xbox_defense:
        try:
            controller = XboxController.open_first(deadband=args.xbox_deadband)
        except RuntimeError as exc:
            print(exc)
            return 2

    print(f"Loaded model: {args.model}")
    if robot_profile is not None:
        print(f"Robot profile: {robot_profile.name}")
    if publisher is not None:
        print(f"NetworkTables server started on 127.0.0.1:{args.port}")
        print("In AdvantageScope: connect to NetworkTables at 127.0.0.1.")
    if custom_ui is not None:
        print(f"Native Field visualizer state: {args.custom_ui_state}")
    if mental_visualizer is not None:
        print("Mental visualizer enabled: add /AdvantageScope/AIAttentionPose and /AI/Mental/*.")
    print("Press Ctrl+C to stop.")

    episode = 0
    try:
        while args.loop or episode < args.episodes:
            obs, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            replay_step = 0
            if publisher is not None:
                publisher.publish(env)
            if custom_ui is not None:
                custom_ui.write(env, [0.0, 0.0, 0.0, 0.0, 0.0, 0.75], 0.0, replay_step, episode, episode_return)
            while True:
                if publisher is not None:
                    publisher.apply_tunables(env)
                if controller is not None:
                    env.set_other_robot_manual_command(*controller.command())
                action, _ = model.predict(
                    _adapt_observation_for_model(model, obs),
                    deterministic=args.deterministic,
                )
                sim_action = action
                if action_adapter is not None:
                    sim_action = action_adapter.adapt(env, action)
                if publisher is not None:
                    publisher.publish_ai_command(sim_action)
                obs, reward, terminated, truncated, info = env.step(sim_action)
                replay_step += 1
                episode_return += float(reward)
                if publisher is not None:
                    publisher.publish(env, reward=float(reward))
                if mental_visualizer is not None:
                    mental_visualizer.publish(
                        build_mental_snapshot(env, sim_action, raw_action=action)
                    )
                if publisher is not None:
                    publisher.publish_training(
                        training_step=0,
                        preview_episode_return=episode_return,
                        preview_episode=episode,
                    )
                if custom_ui is not None:
                    custom_ui.write(env, sim_action, float(reward), replay_step, episode, episode_return)
                time.sleep(max(0.0, env.config.dt_s / max(args.speed, 0.001)))
                if terminated or truncated:
                    print(
                        f"Episode {episode}: return={episode_return:.3f}, "
                        f"scored_coral={info['scored_coral']}"
                    )
                    break
            episode += 1
    except KeyboardInterrupt:
        print("Stopped.")
        if controller is not None:
            controller.close()
        return 0

    if controller is not None:
        controller.close()
    return 0


class CustomUiReplayWriter:
    def __init__(self, state_path: Path, policy_name: str):
        self.state_path = state_path
        self.policy_name = policy_name
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        env: ReefscapeEnv,
        action,
        reward: float,
        replay_step: int,
        episode: int,
        episode_return: float,
    ) -> None:
        payload = build_visualizer_snapshot(
            env,
            action=action,
            reward=reward,
            policy_name=self.policy_name,
            running=True,
        )
        payload["training"] = {
            "step": int(replay_step),
            "previewEpisode": int(episode),
            "previewReturn": float(episode_return),
        }
        tmp_path = self.state_path.with_name(f"{self.state_path.stem}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(8):
            try:
                tmp_path.replace(self.state_path)
                return
            except PermissionError:
                if attempt == 7:
                    break
                time.sleep(0.025 * (attempt + 1))
        try:
            self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)


def _adapt_observation_for_model(model, obs) -> np.ndarray:
    observation = np.asarray(obs, dtype=np.float32)
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not shape or len(shape) != 1:
        return observation

    expected_length = int(shape[0])
    if observation.shape[0] == expected_length:
        return observation
    if observation.shape[0] > expected_length:
        return observation[:expected_length]

    padded = np.zeros(expected_length, dtype=np.float32)
    padded[: observation.shape[0]] = observation
    return padded


if __name__ == "__main__":
    raise SystemExit(main())

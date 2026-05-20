from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.nt_publisher import AdvantageScopeNtPublisher
from reefscape_rl.xbox_controller import XboxController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved PPO model and stream it to AdvantageScope."
    )
    parser.add_argument("--model", type=Path, default=Path("models/reefscape_ppo.zip"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=5810)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--auto-mechanisms", action="store_true")
    parser.add_argument("--manual-mechanisms", action="store_true")
    parser.add_argument("--raw-actions", action="store_true")
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
        print("Install optional dependencies with: pip install -e .[rl]")
        return 2

    args = parse_args()
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return 2

    model = PPO.load(args.model)
    env = ReefscapeEnv(
        ReefscapeEnvConfig(
            randomize_start=not args.fixed_start,
            auto_mechanisms=args.auto_mechanisms and not args.manual_mechanisms,
            other_robot_manual_control=args.xbox_defense,
        )
    )
    action_adapter = None if args.raw_actions else ResidualHeuristicActionAdapter()
    publisher = AdvantageScopeNtPublisher.start_server(port=args.port)
    controller = None
    if args.xbox_defense:
        try:
            controller = XboxController.open_first(deadband=args.xbox_deadband)
        except RuntimeError as exc:
            print(exc)
            return 2

    print(f"Loaded model: {args.model}")
    print(f"NetworkTables server started on 127.0.0.1:{args.port}")
    print("In AdvantageScope: connect to NetworkTables at 127.0.0.1.")
    print("Press Ctrl+C to stop.")

    episode = 0
    try:
        while args.loop or episode < args.episodes:
            obs, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            publisher.publish(env)
            while True:
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
                obs, reward, terminated, truncated, info = env.step(sim_action)
                episode_return += float(reward)
                publisher.publish(env, reward=float(reward))
                publisher.publish_training(
                    training_step=0,
                    preview_episode_return=episode_return,
                    preview_episode=episode,
                )
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

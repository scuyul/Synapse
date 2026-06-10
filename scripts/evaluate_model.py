from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.robot_integration import apply_profile_to_env_config, load_robot_profile
from scripts.run_trained_model import _adapt_observation_for_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--raw-actions", action="store_true")
    parser.add_argument(
        "--robot-profile",
        choices=("sim", "2025-robot"),
        default="sim",
        help="Robot dynamics profile used by the evaluation environment.",
    )
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
    action_adapter = None if args.raw_actions else ResidualHeuristicActionAdapter()
    returns: list[float] = []
    scored: list[int] = []
    hits: list[int] = []
    hard_hits: list[int] = []
    base_config = apply_profile_to_env_config(
        ReefscapeEnvConfig(randomize_start=not args.fixed_start),
        robot_profile,
    )

    for episode in range(args.episodes):
        env = ReefscapeEnv(base_config)
        obs, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        while True:
            action, _ = model.predict(
                _adapt_observation_for_model(model, np.asarray(obs, dtype=np.float32)),
                deterministic=args.deterministic,
            )
            sim_action = action
            if action_adapter is not None:
                sim_action = action_adapter.adapt(env, action)
            obs, reward, terminated, truncated, info = env.step(sim_action)
            episode_return += float(reward)
            if terminated or truncated:
                returns.append(episode_return)
                scored.append(int(info["scored_coral"]))
                hits.append(int(info["other_robot_hits"]))
                hard_hits.append(int(info["other_robot_hard_hits"]))
                break

    print(f"Model: {args.model}")
    if robot_profile is not None:
        print(f"Robot profile: {robot_profile.name}")
    print(f"Episodes: {args.episodes}")
    print(f"Average return: {_mean(returns):.3f}")
    print(f"Average scored coral: {_mean(scored):.3f}")
    print(f"Average robot hits: {_mean(hits):.3f}")
    print(f"Average hard hits: {_mean(hard_hits):.3f}")
    print(f"Best return: {max(returns):.3f}")
    print(f"Best scored coral: {max(scored)}")
    return 0


def _mean(values: list[float] | list[int]) -> float:
    return float(sum(values) / max(1, len(values)))


if __name__ == "__main__":
    raise SystemExit(main())

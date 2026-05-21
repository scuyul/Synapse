from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reefscape_rl.csv_log import AdvantageScopeCsvLogger
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.policies import HeuristicCyclePolicy, RandomPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a REBUILT simulator rollout.")
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("logs/rollout.csv"))
    parser.add_argument("--duration", type=float, default=160.0)
    parser.add_argument("--max-fuel", "--max-coral", dest="max_fuel", type=int, default=100)
    parser.add_argument("--fixed-start", action="store_true")
    return parser.parse_args()


def make_policy(name: str, seed: int):
    if name == "heuristic":
        return HeuristicCyclePolicy()
    return RandomPolicy(seed=seed)


def main() -> int:
    args = parse_args()
    config = ReefscapeEnvConfig(
        episode_duration_s=args.duration,
        max_fuel_scored=args.max_fuel,
        randomize_start=not args.fixed_start,
    )
    env = ReefscapeEnv(config)
    policy = make_policy(args.policy, args.seed)

    total_steps = 0
    episode_returns: list[float] = []
    with AdvantageScopeCsvLogger(args.out) as logger:
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            episode_return = 0.0
            while True:
                action = policy(env)
                _, reward, terminated, truncated, info = env.step(action)
                logger.write_step(env, action, reward, info)
                episode_return += reward
                total_steps += 1
                if terminated or truncated:
                    episode_returns.append(episode_return)
                    break

    avg_return = sum(episode_returns) / len(episode_returns)
    scored = env.state.scored_fuel if env.state is not None else 0
    print(f"Wrote {args.out}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps: {total_steps}")
    print(f"Average return: {avg_return:.3f}")
    print(f"Final episode scored fuel: {scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

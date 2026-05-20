from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.nt_publisher import AdvantageScopeNtPublisher
from reefscape_rl.policies import HeuristicCyclePolicy, RandomPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the REEFSCAPE simulator to AdvantageScope over NetworkTables."
    )
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=5810)
    parser.add_argument("--speed", type=float, default=1.0, help="Realtime playback multiplier.")
    parser.add_argument("--loop", action="store_true", help="Restart when the episode ends.")
    parser.add_argument("--fixed-start", action="store_true")
    return parser.parse_args()


def make_policy(name: str, seed: int):
    if name == "heuristic":
        return HeuristicCyclePolicy()
    return RandomPolicy(seed=seed)


def main() -> int:
    args = parse_args()
    config = ReefscapeEnvConfig(randomize_start=not args.fixed_start)
    env = ReefscapeEnv(config)
    policy = make_policy(args.policy, args.seed)
    publisher = AdvantageScopeNtPublisher.start_server(port=args.port)

    print(f"NetworkTables server started on 127.0.0.1:{args.port}")
    print("In AdvantageScope: connect to NetworkTables at 127.0.0.1.")
    print("Open 2D Field and add /AdvantageScope/RobotPose as a robot pose.")
    print("Add /AdvantageScope/CoralPose and /AdvantageScope/GoalPose as object poses.")
    print("Press Ctrl+C to stop.")

    episode = 0
    try:
        while True:
            env.reset(seed=args.seed + episode)
            publisher.publish(env)
            while True:
                publisher.apply_tunables(env)
                action = policy(env)
                _, reward, terminated, truncated, _ = env.step(action)
                publisher.publish(env, reward=reward)
                sleep_s = max(0.0, env.config.dt_s / max(args.speed, 0.001))
                time.sleep(sleep_s)
                if terminated or truncated:
                    break
            episode += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("Stopped.")
        return 0

    print("Episode complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

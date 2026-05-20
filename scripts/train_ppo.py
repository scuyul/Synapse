from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the REEFSCAPE environment.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", type=Path, default=Path("models/reefscape_ppo"))
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. 'auto' prefers CUDA when PyTorch can see it.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from reefscape_rl.gymnasium_env import GymnasiumReefscapeEnv
    except ImportError as exc:
        print(exc)
        print("Install optional dependencies with: pip install -e .[rl]")
        return 2

    args = parse_args()
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
    env = make_vec_env(lambda: GymnasiumReefscapeEnv(), n_envs=args.n_envs)
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
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_out)
    print(f"Wrote {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

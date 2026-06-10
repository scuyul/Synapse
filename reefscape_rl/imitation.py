"""Heuristic imitation pretraining for PPO policies."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.policies import HeuristicCyclePolicy


@dataclass(slots=True)
class ImitationConfig:
    samples: int = 50_000
    epochs: int = 10
    batch_size: int = 512
    seed: int = 1


def pretrain_from_heuristic(model, config: ImitationConfig) -> None:
    observations, actions = _collect_heuristic_dataset(config)
    device = model.policy.device
    optimizer = model.policy.optimizer
    rng = random.Random(config.seed)

    print(
        f"Heuristic pretraining: {len(observations)} samples, "
        f"{config.epochs} epochs, batch={config.batch_size}"
    )
    for epoch in range(config.epochs):
        indices = list(range(len(observations)))
        rng.shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            batch_obs = torch.as_tensor(
                observations[batch_indices], dtype=torch.float32, device=device
            )
            batch_actions = torch.as_tensor(
                actions[batch_indices], dtype=torch.float32, device=device
            )
            distribution = model.policy.get_distribution(batch_obs)
            predicted_actions = distribution.mode()
            weights = torch.tensor([1.0, 1.0, 0.5, 60.0, 60.0, 8.0], device=device)
            loss = ((predicted_actions - batch_actions).pow(2) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / max(1, len(losses))
        print(f"Heuristic pretrain epoch {epoch + 1}/{config.epochs}: loss={mean_loss:.5f}")

    if hasattr(model.policy, "log_std"):
        model.policy.log_std.data.fill_(-2.0)


def _collect_heuristic_dataset(config: ImitationConfig) -> tuple[np.ndarray, np.ndarray]:
    env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=True))
    policy = HeuristicCyclePolicy()
    rng = random.Random(config.seed)
    observations: list[list[float]] = []
    actions: list[list[float]] = []
    obs, _ = env.reset(seed=config.seed)

    while len(observations) < config.samples:
        action = policy(env)
        repeat = 20 if action[3] > 0.5 or action[4] > 0.5 else 1
        for _ in range(repeat):
            observations.append(obs)
            actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=rng.randrange(1_000_000_000))

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)

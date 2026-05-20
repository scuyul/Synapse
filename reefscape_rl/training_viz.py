"""AdvantageScope visualization callback for PPO training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.nt_publisher import AdvantageScopeNtPublisher


@dataclass(slots=True)
class TrainingVisualizationConfig:
    port: int = 5810
    every_steps: int = 512
    preview_steps: int = 25
    seed: int = 2026


class AdvantageScopeTrainingCallback(BaseCallback):
    """Streams periodic policy rollouts while PPO trains."""

    def __init__(self, config: TrainingVisualizationConfig):
        super().__init__()
        self.config = config
        self.publisher: AdvantageScopeNtPublisher | None = None
        self.preview_env = ReefscapeEnv(
            ReefscapeEnvConfig(randomize_start=False, auto_mechanisms=True)
        )
        self.action_adapter = ResidualHeuristicActionAdapter()
        self.preview_obs: np.ndarray | None = None
        self.preview_episode = 0
        self.preview_return = 0.0

    def _on_training_start(self) -> None:
        self.publisher = AdvantageScopeNtPublisher.start_server(port=self.config.port)
        obs, _ = self.preview_env.reset(seed=self.config.seed)
        self.preview_obs = np.asarray(obs, dtype=np.float32)
        self.publisher.publish(self.preview_env)
        self.publisher.publish_training(
            training_step=0,
            preview_episode_return=0.0,
            preview_episode=self.preview_episode,
        )
        print(f"AdvantageScope training stream started on 127.0.0.1:{self.config.port}")

    def _on_step(self) -> bool:
        if self.publisher is None or self.preview_obs is None:
            return True
        if self.num_timesteps % self.config.every_steps != 0:
            return True

        for _ in range(self.config.preview_steps):
            self.publisher.apply_tunables(self.preview_env)
            action, _ = self.model.predict(self.preview_obs, deterministic=True)
            sim_action = self.action_adapter.adapt(self.preview_env, action)
            obs, reward, terminated, truncated, _ = self.preview_env.step(sim_action)
            self.preview_obs = np.asarray(obs, dtype=np.float32)
            self.preview_return += float(reward)
            self.publisher.publish(self.preview_env, reward=float(reward))
            self.publisher.publish_training(
                training_step=self.num_timesteps,
                preview_episode_return=self.preview_return,
                preview_episode=self.preview_episode,
            )
            if terminated or truncated:
                self.preview_episode += 1
                obs, _ = self.preview_env.reset(seed=self.config.seed + self.preview_episode)
                self.preview_obs = np.asarray(obs, dtype=np.float32)
                self.preview_return = 0.0
                self.publisher.publish(self.preview_env)
                break

        return True

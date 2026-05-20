"""Optional Gymnasium adapter for RL training libraries."""

from __future__ import annotations

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Install optional RL dependencies first: pip install -e .[rl]"
    ) from exc

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import OBSERVATION_FIELDS, ReefscapeEnv, ReefscapeEnvConfig


class GymnasiumReefscapeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ReefscapeEnvConfig | None = None, residual_heuristic: bool = True):
        super().__init__()
        self.env = ReefscapeEnv(config or ReefscapeEnvConfig(auto_mechanisms=True))
        self.action_adapter = ResidualHeuristicActionAdapter() if residual_heuristic else None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_FIELDS),),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        observation, info = self.env.reset(seed=seed)
        return np.asarray(observation, dtype=np.float32), info

    def step(self, action):
        sim_action = action.tolist()
        if self.action_adapter is not None:
            sim_action = self.action_adapter.adapt(self.env, sim_action)
        observation, reward, terminated, truncated, info = self.env.step(sim_action)
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

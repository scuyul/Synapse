"""Optional Gymnasium adapter for RL training libraries."""

from __future__ import annotations

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Install dependencies first: python -m pip install -r .\\requirements.txt"
    ) from exc

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import (
    OBSERVATION_FIELDS,
    OTHER_ROBOT_SPEED_SCALE_MAX,
    ReefscapeEnv,
    ReefscapeEnvConfig,
)


class GymnasiumReefscapeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ReefscapeEnvConfig | None = None,
        residual_heuristic: bool = True,
    ):
        super().__init__()
        self.env = ReefscapeEnv(
            config
            or ReefscapeEnvConfig(
                auto_mechanisms=False,
                randomize_other_robot_start=True,
                randomize_other_robot_behavior=True,
            )
        )
        self.action_adapter = ResidualHeuristicActionAdapter() if residual_heuristic else None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        observation_low, observation_high = _observation_bounds(self.env.config)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
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


def _observation_bounds(config: ReefscapeEnvConfig) -> tuple[np.ndarray, np.ndarray]:
    other_robot_velocity_bound = max(
        1.0,
        _other_robot_velocity_norm_bound(config),
    )
    bounds = {
        "robot_x_norm": (0.0, 1.0),
        "robot_y_norm": (0.0, 1.0),
        "robot_heading_cos": (-1.0, 1.0),
        "robot_heading_sin": (-1.0, 1.0),
        "robot_vx_norm": (-1.0, 1.0),
        "robot_vy_norm": (-1.0, 1.0),
        "robot_omega_norm": (-1.0, 1.0),
        "has_coral": (0.0, 1.0),
        "coral_x_norm": (0.0, 1.0),
        "coral_y_norm": (0.0, 1.0),
        "goal_x_norm": (0.0, 1.0),
        "goal_y_norm": (0.0, 1.0),
        "objective_dx_norm": (-1.0, 1.0),
        "objective_dy_norm": (-1.0, 1.0),
        "objective_distance_norm": (0.0, 1.0),
        "time_remaining_norm": (0.0, 1.0),
        "scored_coral_norm": (0.0, 1.0),
        "other_robot_x_norm": (0.0, 1.0),
        "other_robot_y_norm": (0.0, 1.0),
        "other_robot_heading_cos": (-1.0, 1.0),
        "other_robot_heading_sin": (-1.0, 1.0),
        "other_robot_vx_norm": (
            -other_robot_velocity_bound,
            other_robot_velocity_bound,
        ),
        "other_robot_vy_norm": (
            -other_robot_velocity_bound,
            other_robot_velocity_bound,
        ),
        "other_robot_dx_norm": (-1.0, 1.0),
        "other_robot_dy_norm": (-1.0, 1.0),
        "other_robot_distance_norm": (0.0, 1.0),
    }
    low = np.asarray(
        [bounds[field][0] for field in OBSERVATION_FIELDS],
        dtype=np.float32,
    )
    high = np.asarray(
        [bounds[field][1] for field in OBSERVATION_FIELDS],
        dtype=np.float32,
    )
    return low, high


def _other_robot_velocity_norm_bound(config: ReefscapeEnvConfig) -> float:
    speed_scale = OTHER_ROBOT_SPEED_SCALE_MAX if config.randomize_other_robot_behavior else 1.0
    return max(0.0, config.other_robot_speed_mps) * speed_scale / config.max_linear_speed_mps

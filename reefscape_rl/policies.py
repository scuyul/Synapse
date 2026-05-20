"""Baseline policies for smoke-testing the environment."""

from __future__ import annotations

import math
import random

from reefscape_rl.constants import (
    INTAKE_RADIUS_M,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_SPEED_MPS,
    SCORE_HEADING_TOLERANCE_RAD,
    SCORE_RADIUS_M,
)
from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import clamp, normalize_angle


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def __call__(self, env: ReefscapeEnv) -> list[float]:
        return [
            self._rng.uniform(-1.0, 1.0),
            self._rng.uniform(-1.0, 1.0),
            self._rng.uniform(-1.0, 1.0),
            1.0 if self._rng.random() < 0.05 else 0.0,
            1.0 if self._rng.random() < 0.05 else 0.0,
        ]


class HeuristicCyclePolicy:
    """Simple go-to-objective controller for validating reward and logging."""

    def __call__(self, env: ReefscapeEnv) -> list[float]:
        state = env.state
        if state is None:
            raise RuntimeError("Call env.reset() before using the policy.")

        objective = env.current_objective_pose()
        dx = objective.x - state.pose.x
        dy = objective.y - state.pose.y
        distance = math.hypot(dx, dy)

        vx_norm = clamp((1.8 * dx) / MAX_LINEAR_SPEED_MPS, -1.0, 1.0)
        vy_norm = clamp((1.8 * dy) / MAX_LINEAR_SPEED_MPS, -1.0, 1.0)

        heading_error = normalize_angle(objective.heading - state.pose.heading)
        omega_norm = clamp((3.0 * heading_error) / MAX_ANGULAR_SPEED_RADPS, -1.0, 1.0)

        intake = 0.0
        score = 0.0
        if state.has_coral:
            if distance <= SCORE_RADIUS_M * 0.9 and abs(heading_error) <= SCORE_HEADING_TOLERANCE_RAD:
                score = 1.0
        else:
            if distance <= INTAKE_RADIUS_M * 0.9:
                intake = 1.0

        return [vx_norm, vy_norm, omega_norm, intake, score]

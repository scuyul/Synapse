"""Action adapters for RL policies."""

from __future__ import annotations

from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import clamp
from reefscape_rl.policies import HeuristicCyclePolicy


class ResidualHeuristicActionAdapter:
    """Adds a small learned residual on top of the working heuristic driver."""

    def __init__(self, residual_scale: float = 0.05):
        self.residual_scale = residual_scale
        self.heuristic = HeuristicCyclePolicy()

    def adapt(self, env: ReefscapeEnv, residual_action) -> list[float]:
        base = self.heuristic(env)
        return [
            clamp(base[0] + self.residual_scale * float(residual_action[0]), -1.0, 1.0),
            clamp(base[1] + self.residual_scale * float(residual_action[1]), -1.0, 1.0),
            clamp(base[2] + self.residual_scale * float(residual_action[2]), -1.0, 1.0),
            base[3],
            base[4],
        ]

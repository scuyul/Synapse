from __future__ import annotations

import unittest

try:
    import numpy as np

    from reefscape_rl.constants import MAX_LINEAR_SPEED_MPS
    from reefscape_rl.env import (
        OBSERVATION_FIELDS,
        OTHER_ROBOT_SPEED_SCALE_MAX,
        ReefscapeEnvConfig,
    )
    from reefscape_rl.gymnasium_env import GymnasiumReefscapeEnv
except ImportError as exc:  # pragma: no cover - exercised only without extras
    np = None
    OPTIONAL_IMPORT_ERROR = exc
else:
    OPTIONAL_IMPORT_ERROR = None


class GymnasiumReefscapeEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        if OPTIONAL_IMPORT_ERROR is not None:
            self.skipTest(f"Gymnasium dependencies are not installed: {OPTIONAL_IMPORT_ERROR}")

    def test_observation_space_has_finite_bounds(self) -> None:
        env = GymnasiumReefscapeEnv()

        self.assertEqual(env.observation_space.shape, (len(OBSERVATION_FIELDS),))
        self.assertTrue(np.isfinite(env.observation_space.low).all())
        self.assertTrue(np.isfinite(env.observation_space.high).all())

    def test_reset_and_step_observations_are_inside_declared_space(self) -> None:
        env = GymnasiumReefscapeEnv()
        observation, _ = env.reset(seed=1)

        self.assertTrue(
            env.observation_space.contains(observation),
            _space_violation(env, observation),
        )

        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(50):
            observation, _, terminated, truncated, _ = env.step(action)
            self.assertTrue(
                env.observation_space.contains(observation),
                _space_violation(env, observation),
            )
            if terminated or truncated:
                observation, _ = env.reset(seed=2)
                self.assertTrue(env.observation_space.contains(observation))

    def test_other_robot_velocity_bounds_follow_configured_speed(self) -> None:
        config = ReefscapeEnvConfig(
            other_robot_speed_mps=MAX_LINEAR_SPEED_MPS * 2.0,
            randomize_other_robot_behavior=True,
        )
        env = GymnasiumReefscapeEnv(config=config)
        vx_index = OBSERVATION_FIELDS.index("other_robot_vx_norm")
        vy_index = OBSERVATION_FIELDS.index("other_robot_vy_norm")
        expected = 2.0 * OTHER_ROBOT_SPEED_SCALE_MAX

        self.assertAlmostEqual(env.observation_space.low[vx_index], -expected)
        self.assertAlmostEqual(env.observation_space.high[vx_index], expected)
        self.assertAlmostEqual(env.observation_space.low[vy_index], -expected)
        self.assertAlmostEqual(env.observation_space.high[vy_index], expected)


def _space_violation(env: GymnasiumReefscapeEnv, observation) -> str:
    low = env.observation_space.low
    high = env.observation_space.high
    misses = [
        f"{OBSERVATION_FIELDS[index]}={float(value):.3f} not in "
        f"[{float(low[index]):.3f}, {float(high[index]):.3f}]"
        for index, value in enumerate(observation)
        if value < low[index] or value > high[index]
    ]
    return "; ".join(misses)


if __name__ == "__main__":
    unittest.main()

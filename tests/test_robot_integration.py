from __future__ import annotations

import unittest

from reefscape_rl.env import ReefscapeEnvConfig
from reefscape_rl.robot_integration import (
    apply_profile_to_env_config,
    load_2025_robot_profile,
)


class RobotIntegrationTests(unittest.TestCase):
    def test_loads_2025_robot_drive_profile(self) -> None:
        profile = load_2025_robot_profile()

        self.assertEqual(profile.name, "2025-robot")
        self.assertAlmostEqual(profile.max_linear_speed_mps, 5.22, places=2)
        self.assertGreater(profile.max_angular_speed_radps, 10.0)
        self.assertIn("front_left", profile.module_locations_m)

    def test_profile_updates_env_motion_limits(self) -> None:
        profile = load_2025_robot_profile()
        config = apply_profile_to_env_config(ReefscapeEnvConfig(), profile)

        self.assertEqual(config.max_linear_speed_mps, profile.max_linear_speed_mps)
        self.assertEqual(config.max_angular_speed_radps, profile.max_angular_speed_radps)
        self.assertEqual(config.robot_radius_m, profile.robot_radius_m)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from reefscape_rl.constants import BLUE_CORAL_STATIONS, BLUE_REEF_CENTER
from reefscape_rl.env import OBSERVATION_FIELDS, ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.geometry import Pose2d


class ReefscapeEnvTests(unittest.TestCase):
    def test_reset_returns_expected_observation_size(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        obs, info = env.reset(seed=1)

        self.assertEqual(len(obs), len(OBSERVATION_FIELDS))
        self.assertEqual(info["scored_coral"], 0)
        self.assertFalse(info["has_coral"])

    def test_can_acquire_coral_at_station(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        station = BLUE_CORAL_STATIONS[0]
        env.state.pose = Pose2d(station[0], station[1], 0.0)
        env.state.current_source_index = 0

        _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 1.0, 0.0])

        self.assertGreater(reward, 0.0)
        self.assertFalse(info["has_coral"])
        self.assertEqual(info["event_code"], 3)

        for _ in range(20):
            _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 1.0, 0.0])
            if info["has_coral"]:
                break

        self.assertTrue(info["has_coral"])
        self.assertEqual(info["event_code"], 1)

    def test_can_score_when_at_goal_with_coral(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        goal = env.current_goal_pose()
        env.state.pose = Pose2d(goal.x, goal.y, goal.heading)
        env.state.has_coral = True

        _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 1.0])

        self.assertLess(reward, 1.0)
        self.assertTrue(info["has_coral"])
        self.assertEqual(info["event_code"], 4)

        for _ in range(20):
            _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 1.0])
            if not info["has_coral"]:
                break

        self.assertEqual(info["scored_points"], 5)
        self.assertFalse(info["has_coral"])
        self.assertEqual(info["scored_coral"], 1)
        self.assertEqual(info["event_code"], 2)

    def test_robot_is_pushed_out_of_reef_keepout(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_REEF_CENTER[0], BLUE_REEF_CENTER[1], 0.0)

        _, reward, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertLess(reward, 0.0)
        self.assertGreater(env.state.pose.distance_to(BLUE_REEF_CENTER), 1.5)


if __name__ == "__main__":
    unittest.main()

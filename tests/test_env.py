from __future__ import annotations

import unittest

from reefscape_rl.constants import BLUE_HUB_CENTER, BLUE_TRENCH_LINE_X_M, FUEL_SHOT_RATE_BPS
from reefscape_rl.env import OBSERVATION_FIELDS, ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.geometry import Pose2d


class RebuiltEnvTests(unittest.TestCase):
    def test_reset_returns_expected_observation_size(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        obs, info = env.reset(seed=1)

        self.assertEqual(len(obs), len(OBSERVATION_FIELDS))
        self.assertEqual(info["scored_fuel"], 0)
        self.assertEqual(info["held_fuel"], 0)
        self.assertEqual(info["active_shots"], 0)
        self.assertGreater(info["fuel_pose3d"].z, 0.0)
        self.assertEqual(env.config.robot_fuel_capacity, 50)
        self.assertEqual(info["source_remaining"], 100)
        self.assertTrue(info["hub_active"])
        self.assertEqual(len(env.fuel_poses3d()), 100)

    def test_default_intake_objective_prefers_midfield_fuel(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)

        objective = env.current_objective_pose()

        self.assertGreater(objective.x, 7.0)

    def test_can_drive_over_fuel_to_acquire_it(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        ball = next(ball for ball in env.state.fuel_balls if not ball.collected)
        env.state.pose = Pose2d(ball.x, ball.y, 0.0)

        _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertGreater(reward, 0.0)
        self.assertGreaterEqual(info["held_fuel"], 1)
        self.assertEqual(info["event_code"], 1)
        self.assertLess(info["source_remaining"], 100)

    def test_score_action_launches_fuel_then_scores_in_active_hub(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_TRENCH_LINE_X_M + 0.5, 1.05, 0.0)
        env.state.held_fuel = 1

        for _ in range(20):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
            if info["active_shots"] > 0:
                break

        self.assertEqual(info["held_fuel"], 0)
        self.assertEqual(info["active_shots"], 1)
        self.assertGreater(info["fuel_pose3d"].z, 0.5)
        self.assertEqual(info["event_code"], 4)

        for _ in range(20):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
            if info["scored_fuel"] == 1:
                break

        self.assertEqual(info["scored_points"], 1)
        self.assertEqual(info["scored_fuel"], 1)
        self.assertEqual(info["active_shots"], 0)
        self.assertGreater(info["falling_fuel"], 0)
        self.assertEqual(info["event_code"], 2)

        for _ in range(20):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
            if info["falling_fuel"] == 0:
                break

        self.assertGreaterEqual(info["source_remaining"], 100)

    def test_auto_shoots_at_15_balls_per_second(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_TRENCH_LINE_X_M + 0.5, 1.05, 0.0)
        env.state.held_fuel = 10

        for _ in range(10):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertEqual(info["held_fuel"], 0)
        self.assertAlmostEqual(env.config.shot_period_s, 1.0 / FUEL_SHOT_RATE_BPS)

    def test_inactive_hub_stockpiles_instead_of_shooting(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False, blue_auto_won=True))
        env.reset(seed=1)
        env.state.time_s = 30.0
        env.state.pose = Pose2d(BLUE_TRENCH_LINE_X_M + 0.5, 1.05, 0.0)
        env.state.held_fuel = 1

        for _ in range(20):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertFalse(info["hub_active"])
        self.assertEqual(info["held_fuel"], 1)
        self.assertEqual(info["scored_fuel"], 0)
        self.assertEqual(info["inactive_scored_fuel"], 0)

    def test_cannot_shoot_from_blue_side_of_trench(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_TRENCH_LINE_X_M - 0.5, 1.05, 0.0)
        env.state.held_fuel = 1

        for _ in range(5):
            _, _, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 1.0])

        self.assertEqual(info["held_fuel"], 1)
        self.assertEqual(info["active_shots"], 0)

    def test_trench_blocks_non_corridor_crossing(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_TRENCH_LINE_X_M - 0.15, 4.0, 0.0)

        env.step([1.0, 0.0, 0.0, 0.0, 0.0])

        self.assertLess(env.state.pose.x, BLUE_TRENCH_LINE_X_M)

    def test_robot_is_pushed_out_of_hub_keepout(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        env.state.pose = Pose2d(BLUE_HUB_CENTER[0], BLUE_HUB_CENTER[1], 0.0)

        _, reward, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertLess(reward, 0.0)
        self.assertGreater(env.state.pose.distance_to(BLUE_HUB_CENTER), 1.3)

    def test_freezing_far_from_objective_is_penalized(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)

        rewards = []
        info = {}
        for _ in range(12):
            _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0])
            rewards.append(reward)

        self.assertGreater(info["frozen_time_s"], env.config.freeze_grace_s)
        self.assertLess(rewards[-1], rewards[0])

    def test_smooth_motion_has_better_reward_than_jerky_commands(self) -> None:
        smooth_env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        jerky_env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        smooth_env.reset(seed=1)
        jerky_env.reset(seed=1)

        smooth_total = 0.0
        jerky_total = 0.0
        for index in range(8):
            _, _, _, _, smooth_info = smooth_env.step([0.4, 0.0, 0.0, 0.0, 0.0])
            jerky_action = [1.0 if index % 2 == 0 else -1.0, 0.0, 0.0, 0.0, 0.0]
            _, _, _, _, jerky_info = jerky_env.step(jerky_action)
            smooth_total += smooth_info["smoothness_reward"]
            jerky_total += jerky_info["smoothness_reward"]

        self.assertGreater(smooth_total, jerky_total)


if __name__ == "__main__":
    unittest.main()

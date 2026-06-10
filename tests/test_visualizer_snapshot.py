from __future__ import annotations

import unittest

from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.policies import HeuristicCyclePolicy
from reefscape_rl.visualizer_snapshot import build_visualizer_snapshot


class VisualizerSnapshotTests(unittest.TestCase):
    def test_snapshot_contains_2025_field_and_game_state(self) -> None:
        env = ReefscapeEnv(ReefscapeEnvConfig(randomize_start=False))
        env.reset(seed=1)
        action = HeuristicCyclePolicy()(env)
        _, reward, _, _, _ = env.step(action)

        snapshot = build_visualizer_snapshot(
            env,
            action=action,
            reward=reward,
            policy_name="heuristic",
            running=True,
        )

        self.assertEqual(snapshot["policy"], "heuristic")
        self.assertIn("reefScoringPoses", snapshot["field"])
        self.assertEqual(len(snapshot["field"]["reefScoringPoses"]), 12)
        self.assertEqual(len(snapshot["field"]["coralStations"]), 2)
        self.assertIn(snapshot["match"]["targetLevel"], {"L1", "L2", "L3", "L4"})
        self.assertIn("intent", snapshot["ai"])
        self.assertIn("distanceM", snapshot["objective"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from scripts.train_ppo import (
    TRAIN_UNTIL_STOPPED_TIMESTEPS,
    _resolve_total_timesteps,
)


class TrainPpoArgumentTests(unittest.TestCase):
    def test_zero_timesteps_means_train_until_stopped(self) -> None:
        self.assertEqual(_resolve_total_timesteps(0), TRAIN_UNTIL_STOPPED_TIMESTEPS)

    def test_positive_timesteps_are_unchanged(self) -> None:
        self.assertEqual(_resolve_total_timesteps(100_000), 100_000)

    def test_negative_timesteps_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_total_timesteps(-1)


if __name__ == "__main__":
    unittest.main()

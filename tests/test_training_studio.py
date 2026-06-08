from __future__ import annotations

import unittest

from scripts.training_studio import (
    build_train_command,
    normalize_config,
    validate_launch_config,
)


class TrainingStudioTests(unittest.TestCase):
    def test_build_train_command_includes_core_settings(self) -> None:
        config = normalize_config(
            {
                "timesteps": "256",
                "modelOut": "models/test_model",
                "device": "cpu",
                "nEnvs": "2",
                "nSteps": "64",
                "batchSize": "128",
                "learningRate": "0.001",
                "metricsOut": "logs/test_metrics.jsonl",
            }
        )

        cmd = build_train_command(config)

        self.assertIn("scripts/train_ppo.py", cmd)
        self.assertOptionValue(cmd, "--timesteps", "256")
        self.assertOptionValue(cmd, "--model-out", "models/test_model")
        self.assertOptionValue(cmd, "--device", "cpu")
        self.assertOptionValue(cmd, "--n-envs", "2")
        self.assertOptionValue(cmd, "--metrics-out", "logs/test_metrics.jsonl")

    def test_build_train_command_includes_toggle_flags(self) -> None:
        config = normalize_config(
            {
                "resumeFrom": "models/reefscape_ppo.zip",
                "heuristicPretrain": False,
                "advantageScope": False,
                "variedDefense": False,
            }
        )

        cmd = build_train_command(config)

        self.assertOptionValue(cmd, "--resume-from", "models/reefscape_ppo.zip")
        self.assertIn("--skip-heuristic-pretrain", cmd)
        self.assertIn("--no-advantagescope", cmd)
        self.assertIn("--fixed-defense", cmd)

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_config({"device": "metal"})

    def test_cuda_launch_is_rejected_when_unavailable(self) -> None:
        config = normalize_config({"device": "cuda"})
        compute = {
            "torchInstalled": True,
            "torchVersion": "2.10.0+cpu",
            "cudaAvailable": False,
        }

        with self.assertRaisesRegex(ValueError, "cannot see CUDA"):
            validate_launch_config(config, compute)

    def assertOptionValue(self, cmd: list[str], option: str, expected: str) -> None:
        self.assertIn(option, cmd)
        self.assertEqual(cmd[cmd.index(option) + 1], expected)


if __name__ == "__main__":
    unittest.main()

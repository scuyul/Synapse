# REEFSCAPE RL Simulator

This is the first MVP for training an RL policy to drive a robot in a simplified FRC 2025 REEFSCAPE-style task.

The current simulator is intentionally small:

- One blue-alliance robot on a 2D field.
- Coral-only cycling between coral stations and the reef.
- Continuous action commands: field-relative `vx`, `vy`, `omega`, intake command, score command.
- Filtered observations: robot pose/velocity, held-coral state, coral/source pose, goal pose, time remaining, and score count.
- Reward shaping for progress, acquisition, scoring, and illegal/out-of-bounds behavior.
- live NetworkTables telemetry for AdvantageScope 2D Field visualization.
- CSV logs for fallback debugging outside AdvantageScope.

## Quick Start

Run a heuristic rollout and write a log:

```powershell
python .\scripts\run_rollout.py --policy heuristic --episodes 1 --out .\logs\heuristic_rollout.csv
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Train PPO after installing optional RL dependencies. The trainer defaults to `--device auto`, which uses CUDA when PyTorch can see the RTX 4060:

```powershell
python .\scripts\train_ppo.py --timesteps 100000 --device auto --n-envs 8
```

Training streams a live preview rollout to AdvantageScope by default. While training runs, connect AdvantageScope to NetworkTables at `127.0.0.1` and watch the same `/AdvantageScope/*`, `/Sim/*`, and `/RL/*` topics. Use `--no-advantagescope` to disable this.

Run a saved trained model:

```powershell
python .\scripts\run_trained_model.py --model .\models\reefscape_ppo.zip --fixed-start --loop
```

For this machine, CUDA PyTorch is expected. Verify it with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## AdvantageScope

CSV is not the right path for visualizing this in AdvantageScope. Use the live NetworkTables publisher instead:

```powershell
python .\scripts\live_advantagescope.py --policy heuristic --fixed-start --loop
```

Then in AdvantageScope:

1. Connect to NetworkTables at `127.0.0.1`.
2. Open the `2D Field` tab.
3. Add `/AdvantageScope/RobotPose` as the robot pose.
4. Add `/AdvantageScope/CoralPose`, `/AdvantageScope/GoalPose`, and `/AdvantageScope/ObjectivePose` as object poses.
5. Optionally add `/AdvantageScope/ReefScoringPoses` as a pose array/object set.
6. Plot `/Sim/IsIntaking`, `/Sim/IsScoring`, `/Sim/IntakeProgress`, `/Sim/ScoreProgress`, `/Sim/HasCoral`, and `/Sim/ScoredCoral` to see pickup/placement timing.
7. Tune `/Tuning/IntakeDurationS` and `/Tuning/ScoreDurationS` live in NetworkTables. Both default to `0.25`.
8. During RL training, plot `/RL/TrainingStep`, `/RL/PreviewEpisodeReturn`, and `/RL/PreviewEpisode`.

The CSV logger still exists for quick plots/debugging outside AdvantageScope:

- `/Sim/RobotPose/x`
- `/Sim/RobotPose/y`
- `/Sim/RobotPose/heading`
- `/Sim/CoralPose/x`
- `/Sim/CoralPose/y`
- `/Sim/GoalPose/x`
- `/Sim/GoalPose/y`
- `/RL/Reward`
- `/RL/TotalReward`
- `/RL/Action/*`

## Next Milestones

1. Add a Gymnasium wrapper and Stable-Baselines3 PPO training script.
2. Add WPILOG/NT4 structured telemetry for native AdvantageScope Field visualization.
3. Add algae, processor/net scoring, branch occupancy, and optional opponent robots.
4. Add domain randomization for sensor noise, latency, friction, and start poses.
5. Add a policy safety wrapper that constrains outputs before deployment to robot code.

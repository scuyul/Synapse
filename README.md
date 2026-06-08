# REEFSCAPE RL Simulator

This is a prototype for training an RL policy to drive a robot in a simplified FRC 2025 REEFSCAPE-style task.

The current simulator is intentionally small:

- One controlled blue-alliance robot plus one moving traffic robot on the blue side.
- Coral-only cycling between coral stations and the reef.
- Continuous action commands: field-relative `vx`, `vy`, `omega`, intake command, score command.
- Filtered observations: robot pose/velocity, held-coral state, coral/source pose, goal pose, moving robot pose/velocity, time remaining, and score count.
- Reward shaping for progress, smooth driving, acquisition, scoring, illegal/out-of-bounds behavior, reef collisions, low-speed taps, hard robot-to-robot impacts, and freezing far from the objective.
- live NetworkTables telemetry for AdvantageScope 2D Field visualization.
- CSV logs for fallback debugging outside AdvantageScope.

## Quick Start

Prerequisites:

- Windows with Python 3.11 or newer.
- An NVIDIA GPU/driver for CUDA training. This project defaults to CUDA and uses CPU only as a fallback.
- AdvantageScope is optional but recommended for live field visualization.

Create a local virtual environment and install the pinned CUDA runtime:

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
reefscape-doctor
```

Use the interactive menu:

```powershell
python .\menu.py
```

Open the browser-based training studio directly:

```powershell
python .\scripts\training_studio.py
```

Run a heuristic rollout and write a log:

```powershell
python .\scripts\run_rollout.py --policy heuristic --episodes 1 --out .\logs\heuristic_rollout.csv
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Development

Generated training outputs are intentionally not tracked by git:

- `logs/`
- `models/`
- `.venv/`
- Python caches and build artifacts

Before committing changes, run:

```powershell
python -m unittest discover -s tests
python -m compileall reefscape_rl scripts tests
reefscape-doctor
```

If CUDA is unavailable on a development machine, use `--device auto` or
`--device cpu` for smoke tests, but keep CUDA as the default path.

Train PPO after installing optional RL dependencies. The trainer defaults to `--device cuda` for the RTX 4060; use `--device auto` or `--device cpu` only as a fallback:

```powershell
python .\scripts\train_ppo.py --timesteps 100000 --device cuda --n-envs 8
```

Use `--timesteps 0` to train until you stop it with Ctrl+C. The trainer will
save the current model to `*_interrupted.zip` when interrupted.

Training randomizes the defense bot's start phase, direction, path variant, and speed by default so the policy is better prepared for Xbox-controlled defense. Use `--fixed-defense` only when you want the old repeatable defense path.

Training streams a live preview rollout to AdvantageScope by default. While training runs, connect AdvantageScope to NetworkTables at `127.0.0.1` and watch the same `/AdvantageScope/*`, `/Sim/*`, and `/RL/*` topics. Use `--no-advantagescope` to disable this.

New models train as residual controllers on top of the working heuristic pathing driver. The RL policy learns corrections, while the baseline prevents jitter and keeps the robot moving toward valid targets. The moving traffic robot only affects reward/physics on body contact, so the policy can choose close passes instead of taking large detours. Intake/score commands come from the policy wrapper by default; use `--auto-mechanisms` only for experiments where you want the environment to trigger mechanisms automatically. Use `--raw-actions` when running a model only if you intentionally trained a fully raw policy.

Run a saved trained model:

```powershell
python .\scripts\run_trained_model.py --model .\models\reefscape_ppo.zip --fixed-start --loop
```

For demo mode, add the synthetic mental visualizer. It publishes readable
`/AI/Mental/*` telemetry plus `/AdvantageScope/AIAttentionPose`, so the field can
show what the policy is focusing on while plots show intent, confidence, risk,
and action energy:

```powershell
python .\scripts\run_trained_model.py --model .\models\reefscape_ppo.zip --fixed-start --loop --mental-visualizer
```

To manually drive the defense robot with an Xbox controller, connect the controller first and opt in:

```powershell
python .\scripts\run_trained_model.py --model .\models\reefscape_ppo.zip --fixed-start --loop --xbox-defense
```

Left stick drives the defense robot field-relative. Right stick X rotates it. This is off by default.

Models trained before the moving traffic robot was added can still replay through the compatibility adapter, but they did not learn the new obstacle observations. Retrain for real collision avoidance behavior.

Training saves rotating checkpoints in `models/checkpoints` by default and keeps the latest two. If you press Ctrl+C during training, it saves `models/reefscape_ppo_interrupted.zip`.

Resume training:

```powershell
python .\scripts\train_ppo.py --resume-from .\models\reefscape_ppo_interrupted.zip --timesteps 100000 --device cuda
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
4. Add `/AdvantageScope/OtherRobotPose`, `/AdvantageScope/CoralPose`, `/AdvantageScope/GoalPose`, and `/AdvantageScope/ObjectivePose` as object poses.
5. Optionally add `/AdvantageScope/ReefScoringPoses` as a pose array/object set.
6. Plot `/Sim/IsIntaking`, `/Sim/IsScoring`, `/Sim/IntakeProgress`, `/Sim/ScoreProgress`, `/Sim/HasCoral`, `/Sim/ScoredCoral`, `/Sim/FrozenTime`, `/RL/SmoothnessReward`, `/Sim/OtherRobotDistance`, `/Sim/HitOtherRobot`, `/Sim/HardHitOtherRobot`, `/Sim/OtherRobotHits`, `/Sim/OtherRobotHardHits`, and `/Sim/OtherRobotImpactSpeed` to see pickup/placement timing, stalls, smoothness, and collision severity.
7. Tune `/Tuning/IntakeDurationS` and `/Tuning/ScoreDurationS` live in NetworkTables. Both default to `0.25`.
8. During RL training, plot `/RL/TrainingStep`, `/RL/PreviewEpisodeReturn`, and `/RL/PreviewEpisode`.
9. When running a trained model with `--mental-visualizer`, add `/AdvantageScope/AIAttentionPose` as an object pose and plot `/AI/Mental/Intent`, `/AI/Mental/Focus`, `/AI/Mental/Confidence`, `/AI/Mental/Risk`, `/AI/Mental/ActionEnergy`, and `/AI/Mental/ThoughtTrace`.

The CSV logger still exists for quick plots/debugging outside AdvantageScope:

- `/Sim/RobotPose/x`
- `/Sim/RobotPose/y`
- `/Sim/RobotPose/heading`
- `/Sim/CoralPose/x`
- `/Sim/CoralPose/y`
- `/Sim/GoalPose/x`
- `/Sim/GoalPose/y`
- `/Sim/OtherRobotPose/x`
- `/Sim/OtherRobotPose/y`
- `/RL/Reward`
- `/RL/TotalReward`
- `/RL/Action/*`

## Next Milestones

1. Add a repeatable PPO evaluation report with scoring, collision, and cycle-time metrics.
2. Add WPILOG/NT4 structured telemetry for native AdvantageScope Field visualization.
3. Add algae, processor/net scoring, branch occupancy, and richer multi-robot traffic.
4. Add domain randomization for sensor noise, latency, friction, and start poses.
5. Add a policy safety wrapper that constrains outputs before deployment to robot code.

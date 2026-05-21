# REBUILT RL Simulator

This is a compact RL simulator for the FRC 2026 REBUILT game. It keeps the original Python package name for compatibility, but the environment now models FUEL cycling into the blue HUB instead of 2025 coral-to-reef scoring.

The current simulator is intentionally small:

- One controlled blue-alliance robot.
- 100 individual FUEL balls start in the middle of the field.
- Driving over FUEL intakes it automatically.
- Integer held-FUEL inventory with a default 50-FUEL robot capacity.
- The trench line is enforced; the robot can only move between the alliance side and midfield through the trench corridors.
- The robot auto-shoots from the far/midfield side of the trench line at 15 FUEL per second during active HUB windows.
- Continuous action commands: field-relative `vx`, `vy`, `omega`, intake command, score command.
- REBUILT match timing: 20s AUTO, 10s transition, four 25s alliance shifts, and 30s END GAME.
- Blue HUB active/inactive scoring windows based on the configured AUTO result.
- Simple 3D FUEL shot pose/arc after the score command launches a ball toward the HUB.
- Filtered observations: robot pose/velocity, held FUEL, source/goal/objective pose, active HUB state, match phase, active shot count, time remaining, and score count.
- Reward shaping for progress, smooth driving, acquisition, shot launches, active-HUB scoring, stockpiling during inactive windows, illegal/out-of-bounds behavior, HUB collisions, missed shots, and freezing far from the objective.
- Live NetworkTables telemetry for AdvantageScope 2D Field visualization.
- CSV logs for fallback debugging outside AdvantageScope.

## Quick Start

Use the interactive menu:

```powershell
python .\menu.py
```

Run a heuristic rollout and write a log:

```powershell
python .\scripts\run_rollout.py --policy heuristic --episodes 1 --out .\logs\heuristic_rollout.csv
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Train PPO after installing optional RL dependencies:

```powershell
python .\scripts\train_ppo.py --timesteps 100000 --device auto --n-envs 8
```

Run a saved trained model:

```powershell
python .\scripts\run_trained_model.py --model .\models\rebuilt_ppo.zip --fixed-start --loop
```

## AdvantageScope

Use the live NetworkTables publisher:

```powershell
python .\scripts\live_advantagescope.py --policy heuristic --fixed-start --loop
```

Then in AdvantageScope:

1. Connect to NetworkTables at `127.0.0.1`.
2. Open the `2D Field` tab.
3. Add `/AdvantageScope/RobotPose` as the robot pose.
4. Add `/AdvantageScope/FuelPose3d`, `/AdvantageScope/FuelPoses3d`, `/AdvantageScope/GoalPose`, and `/AdvantageScope/ObjectivePose` as object poses.
5. Optionally add `/AdvantageScope/HubScoringPoses` as a pose array/object set.
6. Plot `/Sim/HeldFuel`, `/Sim/ScoredFuel`, `/Sim/InactiveScoredFuel`, `/Sim/MissedFuel`, `/Sim/ActiveShots`, `/Sim/HubActive`, `/Sim/MatchPhase`, `/Sim/IsIntaking`, `/Sim/IsScoring`, `/Sim/IntakeProgress`, `/Sim/ScoreProgress`, and `/RL/SmoothnessReward` for debugging. `/AdvantageScope/FuelPoses3d` is a single Pose3d array topic containing all visible FUEL.
7. Tune `/Tuning/IntakeDurationS` and `/Tuning/ScoreDurationS` live in NetworkTables.

## Scope

This is still a simplified training environment. It does not yet simulate exact field CAD, tower climbs, human-player scoring, or full alliance strategy.

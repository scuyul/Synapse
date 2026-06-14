# Synapse / REEFSCAPE RL

Note: This was made by over-caffeinated highschoolers and chatgpt things are: bad, broken, and violently vibe coded. Enter at your own risk

Synapse is a native Windows desktop app for a simplified FRC 2025 REEFSCAPE-style RL simulator. The app handles setup, training, model playback, live metrics, logs, and the built-in field visualizer in one place. The command line tools still exist, but the desktop app is the main way to use this now.

## What it does

- Trains PPO policies for a blue-alliance REEFSCAPE driving task.
- Uses a built-in field visualizer by default, so you do not need a web UI.
- Can host NetworkTables data for AdvantageScope if you want to use it.
- Shows training status, reward metrics, logs, model artifacts, and live field state in the app.
- Keeps the CLI available for fallback/debugging.

The simulator currently includes:

- One controlled blue-alliance robot plus one moving traffic robot.
- Coral-only cycling between coral stations and the reef.
- Field-relative drive actions: `vx`, `vy`, `omega`, intake, and score.
- Observations for robot pose/velocity, coral state, source/goal pose, traffic robot pose/velocity, time, and score count.
- Reward shaping for progress, smooth driving, pickup, scoring, bounds, reef collisions, traffic hits, and getting stuck.

## Install

The normal install is the Windows installer from the release:

1. Download `ReefscapeRL-Setup.exe` from the latest release.
2. Run it.
3. Open `Reefscape RL` from the Start Menu.
4. In the app, click `Setup + Deps` the first time you use it.
5. Click `Doctor` to make sure Python/CUDA/dependencies are good.
6. Use `Train`, `Field`, `Models`, and `Logs` from the sidebar.

If you do not want to install it, use the portable build:

1. Download `ReefscapeRL-portable.zip`.
2. Extract it somewhere normal, like `Documents` or `Desktop`.
3. Run `ReefscapeRL.exe` inside the extracted folder.
4. Run `Setup + Deps` once from inside the app.

Do not run the exe from inside the zip file. Extract it first.

## Training

Open the desktop app and go to `Train`.

Good default settings:

- `Device`: `cuda`
- `Timesteps`: `100000`
- `Parallel Envs`: `8`
- `Preview`: `desktop`
- `Model Out`: `models/reefscape_ppo`

Click `Train`. The app will show current status, live reward metrics, logs, and the field visualizer. If CUDA is not available, use `auto` or `cpu`, but CUDA is the main path.

Models are saved under `models/`. Training runs and metrics are saved under `runs/` and `logs/`.

## Running a model

Open the desktop app and go to `Models`.

1. Pick a `.zip` model.
2. Click `Run in Field`.
3. The model will run in the built-in field visualizer.
4. Use the global `Stop` button to stop it.

Older models may still replay through compatibility code, but if the simulator changed, retraining is usually better.

## AdvantageScope

AdvantageScope is optional. The app does not launch AdvantageScope for you anymore.

Use the `Host AdvantageScope` button in the desktop app. That starts the local NetworkTables sim stream. Then open AdvantageScope yourself and connect to:

```text
127.0.0.1
```

Useful topics:

- `/AdvantageScope/RobotPose`
- `/AdvantageScope/OtherRobotPose`
- `/AdvantageScope/CoralPose`
- `/AdvantageScope/GoalPose`
- `/AdvantageScope/ObjectivePose`
- `/AdvantageScope/ReefScoringPoses`
- `/RL/TrainingStep`
- `/RL/PreviewEpisodeReturn`
- `/RL/Reward`
- `/Sim/HasCoral`
- `/Sim/ScoredCoral`

The built-in visualizer is still the default path. AdvantageScope is just there if you want it.

## Building from source

Prerequisites:

- Windows
- Python 3.11 or newer
- Node.js/npm
- Go
- Wails
- NVIDIA driver/CUDA if you want fast training
- Inno Setup if you want the Windows installer

From the repo root:

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
reefscape-doctor
python build.py --clean --installer
```

Build outputs go to `builds/`:

- `builds\ReefscapeRL\ReefscapeRL.exe`
- `builds\ReefscapeRL-portable.zip`
- `builds\ReefscapeRL-Setup.exe`

For a portable-only build:

```powershell
python build.py --clean
```

## CLI fallback

The app is the main UI, but the command line tools are still useful for debugging.

Setup:

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
reefscape-doctor
```

Train:

```powershell
python .\scripts\train_ppo.py --timesteps 100000 --device cuda --n-envs 8 --visualization-backend custom-ui
```

Run a saved model:

```powershell
python .\scripts\run_trained_model.py --model .\models\reefscape_ppo.zip --fixed-start --loop --visualization-backend custom-ui
```

Run heuristic rollout:

```powershell
python .\scripts\run_rollout.py --policy heuristic --episodes 1 --out .\logs\heuristic_rollout.csv
```

Old menus:

```powershell
python .\menu.py
python .\manage.py
```

## Development checks

Before pushing code changes:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests
cd app
go test ./...
cd frontend
npm run build
```

## Git notes

Generated files are not tracked:

- `.venv/`
- `logs/`
- `models/`
- `runs/`
- `builds/`
- `dist/`
- `node_modules/`
- Python caches
- robot project build output

Release files should be uploaded to GitHub Releases, not committed into git.

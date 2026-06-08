# Contributing

Thanks for helping improve REEFSCAPE RL.

## Local setup

Use a virtual environment instead of installing into global Python:

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
reefscape-doctor
```

## Checks

Run these before opening a pull request:

```powershell
python -m unittest discover -s tests
python -m compileall reefscape_rl scripts tests
```

## Generated files

Do not commit generated models, logs, virtual environments, caches, or local
environment files. The repository `.gitignore` excludes the expected generated
paths.

## Project direction

Keep simulator changes deterministic when possible, add tests for reward or
observation contract changes, and prefer small focused pull requests.

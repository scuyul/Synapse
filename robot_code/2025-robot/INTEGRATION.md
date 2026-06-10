# 2025 Robot Code Snapshot

This directory is a source snapshot copied from:

`C:\Users\Scuyul\Documents\GitHub\2025-robot`

Included:

- `src/` robot source and deploy files
- `vendordeps/`
- Gradle wrapper/build files
- dashboard/sim config files
- team utility folders that were present in the source project

Excluded:

- source Git history
- `.gradle/`, `build/`, `bin/`
- local IDE state
- large AdvantageKit `.glb` visualization models

The simulator reads drivetrain constants from:

`src/main/java/frc/robot/Robot25/subsystems/drive/DriveConstants.java`

Use the `2025-robot` training profile when you want RL training to use the copied robot drivetrain speed, wheel radius, module geometry, and derived angular-speed limit. Existing models trained on the old simulator profile should be treated as incompatible unless you intentionally validate or retrain them.

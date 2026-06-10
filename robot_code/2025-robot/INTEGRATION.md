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

## AI Driver Hold Mode

`driverController.back()` is wired to hold-to-run `AiModeCommand`.

While held, the robot reads these NetworkTables entries:

- `/AI/Command/vx`: field-relative normalized X command, `-1.0..1.0`
- `/AI/Command/vy`: field-relative normalized Y command, `-1.0..1.0`
- `/AI/Command/omega`: normalized angular command, `-1.0..1.0`
- `/AI/Command/intake`: request intake when `> 0.5`
- `/AI/Command/score`: request score when `> 0.5`
- `/AI/Command/scoreLevel`: optional requested reef level `1..4`, defaults to `4`
- `/AI/Command/enabled`: set true by the robot while AI mode is active

The intake request schedules the existing robot intake path:

`elevator.intakeHeight().andThen(outtake.autoQueueCoral3())`

The score request schedules the existing robot scoring path:

`elevator.Lx().andThen(outtake.depositCoral()).andThen(elevator.intakeHeight())`

`Lx` is selected from `/AI/Command/scoreLevel`; if no level is published, AI scoring defaults to L4.

Releasing Back ends AI mode, stops the drivetrain, and normal driver default control resumes.

# MVP Assumptions

This simulator is not a full official field simulator yet. It is a training scaffold.

## Official Inputs Used

- REEFSCAPE field size is approximated from the official 2025 manual: about 57 ft 6 7/8 in by 26 ft 5 in, represented here as `17.55 m x 8.05 m`.
- Match duration is represented as 150 seconds.
- Teleop coral scoring values are represented as L1 = 2, L2 = 3, L3 = 4, L4 = 5.
- The current MVP targets L4 by default.

## Simplifications

- Only one blue-alliance robot is simulated.
- Coral stations are simplified as source points instead of full field assemblies.
- Reef branches are approximated as 12 scoring poses around a circular reef center.
- Coral acquisition and scoring use radius checks instead of mechanism geometry.
- Coral pickup takes `0.25 s` of continuous intake action while in range and settled.
- Coral placement takes `0.25 s` of continuous score action while aligned, in range, and settled.
- Pickup and placement durations are tunable live through `/Tuning/IntakeDurationS` and `/Tuning/ScoreDurationS`.
- The reef is modeled as a circular keepout obstacle that pushes the robot out and applies a collision penalty.
- Algae, processor, net, barge, cages, penalties, opponents, and exact protected-zone rules are not simulated yet.
- AdvantageScope visualization uses live NT4 struct telemetry from `scripts/live_advantagescope.py`.
- CSV logs are fallback/debug artifacts, not the main AdvantageScope visualization path.

## Near-Term Design Direction

The right next step is to add WPILOG recording of the same struct telemetry so trained/evaluation episodes can be opened later without replaying a live NetworkTables stream.

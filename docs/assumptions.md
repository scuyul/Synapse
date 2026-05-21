# MVP Assumptions

This simulator is not a full official field simulator yet. It is a training scaffold for the 2026 REBUILT game.

## Official Inputs Used

- REBUILT uses FUEL as the single scoring element.
- FUEL scored in an active HUB is worth 1 point in AUTO and TELEOP; FUEL scored in an inactive HUB earns no points.
- MATCH timing is represented as 160 seconds: 20 seconds AUTO, 10 seconds TRANSITION SHIFT, four 25-second ALLIANCE SHIFTS, and 30 seconds END GAME.
- Both HUBS are active during AUTO, TRANSITION SHIFT, and END GAME. During ALLIANCE SHIFTS, the blue HUB alternates active/inactive based on the configured AUTO result.
- The field is approximated as `16.54 m x 8.07 m`.

## Simplifications

- One controlled blue-alliance robot is simulated.
- 100 FUEL balls are initialized in a deterministic midfield grid.
- Driving over a FUEL ball intakes it automatically when the robot has capacity.
- The robot can hold up to 50 FUEL.
- A simplified trench line separates the blue alliance side from midfield, and crossing is only allowed through the top/bottom trench corridors.
- The robot auto-shoots held FUEL from the far/midfield side of the trench line at 15 balls per second during active HUB windows.
- During inactive HUB windows, the heuristic keeps collecting FUEL instead of shooting it.
- The robot tracks an integer held-FUEL count and creates simple 3D in-flight FUEL poses after scoring.
- The blue HUB is approximated as a circular keepout obstacle with several scoring poses on the alliance-zone side.
- FUEL acquisition uses radius checks; HUB scoring launches a simple constant-velocity shot toward the HUB.
- FUEL pickup takes `0.18 s` of continuous intake action while in range and settled.
- FUEL scoring takes `0.12 s` of continuous score action while aligned, in range, and settled.
- Pickup and scoring durations are tunable live through `/Tuning/IntakeDurationS` and `/Tuning/ScoreDurationS`.
- Tower climbs, human-player scoring, exact bump/trench geometry, exact AprilTag layout, full foul logic, and full alliance strategy are not simulated yet.
- Freezing far from the current objective is explicitly penalized after a short grace period; mechanism pauses are exempt.
- Smooth motion shaping rewards consistent movement and penalizes rapid command changes; it does not reward sitting still.
- AdvantageScope visualization uses live NT4 struct telemetry from `scripts/live_advantagescope.py`.
- CSV logs are fallback/debug artifacts, not the main AdvantageScope visualization path.

## Near-Term Design Direction

The right next step is to add tower/endgame actions and WPILOG recording of the same struct telemetry so trained/evaluation episodes can be opened later without replaying a live NetworkTables stream.

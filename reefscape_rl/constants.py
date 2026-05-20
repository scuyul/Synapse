"""Constants for the simplified REEFSCAPE simulator.

Field dimensions are based on the official 2025 REEFSCAPE manual's nominal
dimensions: approximately 57 ft 6 7/8 in by 26 ft 5 in.
"""

from __future__ import annotations

import math

FIELD_LENGTH_M = 17.55
FIELD_WIDTH_M = 8.05
FIELD_DIAGONAL_M = math.hypot(FIELD_LENGTH_M, FIELD_WIDTH_M)
HALF_FIELD_LENGTH_M = FIELD_LENGTH_M / 2.0

MATCH_DURATION_S = 150.0
CONTROL_PERIOD_S = 0.1

ROBOT_RADIUS_M = 0.48
MAX_LINEAR_SPEED_MPS = 4.5
MAX_ANGULAR_SPEED_RADPS = 7.0
MAX_LINEAR_ACCEL_MPS2 = 7.0
MAX_ANGULAR_ACCEL_RADPS2 = 14.0

INTAKE_RADIUS_M = 0.7
SCORE_RADIUS_M = 0.85
SCORE_HEADING_TOLERANCE_RAD = math.radians(75.0)

BLUE_REEF_CENTER = (4.50, FIELD_WIDTH_M / 2.0)
REEF_SCORING_RADIUS_M = 1.55

BLUE_CORAL_STATIONS = (
    (0.85, 1.15),
    (0.85, FIELD_WIDTH_M - 1.15),
)

SCORING_POINTS_TELEOP = {
    "L1": 2,
    "L2": 3,
    "L3": 4,
    "L4": 5,
}

SCORING_POINTS_AUTO = {
    "L1": 3,
    "L2": 4,
    "L3": 6,
    "L4": 7,
}

DEFAULT_TARGET_LEVEL = "L4"
DEFAULT_MAX_CORAL_SCORED = 12

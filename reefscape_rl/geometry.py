"""Small geometry helpers used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class Pose2d:
    x: float
    y: float
    heading: float

    def distance_to(self, other: "Pose2d | tuple[float, float]") -> float:
        if isinstance(other, Pose2d):
            ox = other.x
            oy = other.y
        else:
            ox, oy = other
        return math.hypot(self.x - ox, self.y - oy)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_to(src_x: float, src_y: float, dst_x: float, dst_y: float) -> float:
    return math.atan2(dst_y - src_y, dst_x - src_x)


def approach(current: float, target: float, max_delta: float) -> float:
    delta = clamp(target - current, -max_delta, max_delta)
    return current + delta

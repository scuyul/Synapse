"""Baseline policies for smoke-testing the REBUILT environment."""

from __future__ import annotations

import math
import random

from reefscape_rl.constants import (
    BLUE_REEF_CENTER,
    BLUE_MIDFIELD_SWEEP_POINTS,
    BLUE_TRENCH_LINE_X_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    INTAKE_RADIUS_M,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_SPEED_MPS,
    REEF_CLEARANCE_M,
    REEF_OBSTACLE_RADIUS_M,
    ROBOT_RADIUS_M,
    TRENCH_CORRIDOR_WIDTH_M,
)
from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import clamp, normalize_angle


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def __call__(self, env: ReefscapeEnv) -> list[float]:
        return [
            self._rng.uniform(-1.0, 1.0),
            self._rng.uniform(-1.0, 1.0),
            self._rng.uniform(-1.0, 1.0),
            1.0 if self._rng.random() < 0.05 else 0.0,
            1.0 if self._rng.random() < 0.05 else 0.0,
        ]


class HeuristicCyclePolicy:
    """Simple go-to-objective controller for validating reward and logging."""

    def __call__(self, env: ReefscapeEnv) -> list[float]:
        state = env.state
        if state is None:
            raise RuntimeError("Call env.reset() before using the policy.")

        objective = env.current_objective_pose()
        objective_distance = state.pose.distance_to(objective)
        sweep_target = self._midfield_sweep_waypoint(env)
        source_return_target = self._source_return_waypoint(env, objective.x, objective.y)
        scoring_route_target = self._scoring_route_waypoint(env, objective.x, objective.y)
        if sweep_target is not None:
            target_x, target_y = sweep_target
        elif source_return_target is not None:
            target_x, target_y = source_return_target
        elif scoring_route_target is not None:
            target_x, target_y = scoring_route_target
        elif state.has_fuel and objective_distance < 2.0:
            target_x, target_y = objective.x, objective.y
        else:
            target_x, target_y = self._avoid_reef_waypoint(
                state.pose.x,
                state.pose.y,
                objective.x,
                objective.y,
            )
        dx = target_x - state.pose.x
        dy = target_y - state.pose.y
        target_distance = math.hypot(dx, dy)

        vx_norm = clamp((1.8 * dx) / MAX_LINEAR_SPEED_MPS, -1.0, 1.0)
        vy_norm = clamp((1.8 * dy) / MAX_LINEAR_SPEED_MPS, -1.0, 1.0)
        if target_distance > 0.25:
            vx_norm = _with_min_command(vx_norm, 0.08)
            vy_norm = _with_min_command(vy_norm, 0.08)

        heading_error = normalize_angle(objective.heading - state.pose.heading)
        omega_norm = clamp((3.0 * heading_error) / MAX_ANGULAR_SPEED_RADPS, -1.0, 1.0)

        intake = 0.0
        score = 0.0
        should_stockpile = (
            state.has_fuel
            and not env.is_blue_hub_active()
            and state.held_fuel < env.config.robot_fuel_capacity
        )
        if state.has_fuel and not should_stockpile:
            if (
                env.is_blue_hub_active()
                and state.pose.x >= BLUE_TRENCH_LINE_X_M
                and self._is_settled(env)
            ):
                vx_norm = 0.0
                vy_norm = 0.0
                omega_norm = 0.0
                score = 1.0
        else:
            fuel_pose = env.current_fuel_pose()
            can_intake = (
                state.pose.distance_to((fuel_pose.x, fuel_pose.y)) <= INTAKE_RADIUS_M
                and self._is_settled(env)
            )
            if can_intake:
                vx_norm = 0.0
                vy_norm = 0.0
                omega_norm = 0.0
                intake = 1.0

        return [vx_norm, vy_norm, omega_norm, intake, score]

    def _is_settled(self, env: ReefscapeEnv) -> bool:
        state = env.state
        if state is None:
            return False
        return math.hypot(state.vx_mps, state.vy_mps) < 0.25 and abs(state.omega_radps) < 0.5

    def _source_return_waypoint(
        self, env: ReefscapeEnv, source_x: float, source_y: float
    ) -> tuple[float, float] | None:
        state = env.state
        if state is None or state.has_fuel:
            return None
        if source_x > FIELD_LENGTH_M * 0.4:
            return None

        center_x, center_y = BLUE_REEF_CENTER
        if state.pose.x < center_x - 1.7:
            return None

        top_side = source_y > center_y
        lane_y = FIELD_WIDTH_M - 1.35 if top_side else 1.35
        exit_x = center_x - 1.85
        exit_y = center_y + 2.35 if top_side else center_y - 2.35

        if abs(state.pose.y - lane_y) < 0.45 and state.pose.x < center_x - 0.5:
            return None
        if state.pose.x > center_x - 0.2:
            return (state.pose.x - 0.8, exit_y)
        return (exit_x, lane_y)

    def _midfield_sweep_waypoint(self, env: ReefscapeEnv) -> tuple[float, float] | None:
        state = env.state
        if state is None:
            return None
        if state.has_fuel and (
            env.is_blue_hub_active() or state.held_fuel >= env.config.robot_fuel_capacity
        ):
            return None

        objective = env.current_objective_pose()
        midfield_x = FIELD_LENGTH_M / 2.0
        if state.pose.x < BLUE_TRENCH_LINE_X_M - 0.35 and not _is_in_trench_corridor(state.pose.y):
            lane_y = 1.05 if objective.y < FIELD_WIDTH_M / 2.0 else FIELD_WIDTH_M - 1.05
            return BLUE_TRENCH_LINE_X_M - 0.65, lane_y
        if state.pose.x < midfield_x - 2.0:
            return objective.x, objective.y

        index = int(state.time_s // 1.2) % len(BLUE_MIDFIELD_SWEEP_POINTS)
        return BLUE_MIDFIELD_SWEEP_POINTS[index]

    def _scoring_route_waypoint(
        self, env: ReefscapeEnv, goal_x: float, goal_y: float
    ) -> tuple[float, float] | None:
        state = env.state
        if state is None or not state.has_fuel or not env.is_blue_hub_active():
            return None

        center_x, center_y = BLUE_REEF_CENTER
        lane_y = FIELD_WIDTH_M - 1.05 if state.pose.y > center_y else 1.05
        far_entry = (BLUE_TRENCH_LINE_X_M + 0.65, lane_y)
        if state.pose.x < BLUE_TRENCH_LINE_X_M - 0.20 and state.pose.distance_to((BLUE_TRENCH_LINE_X_M - 0.65, lane_y)) > 0.30:
            return BLUE_TRENCH_LINE_X_M - 0.65, lane_y
        if state.pose.x < BLUE_TRENCH_LINE_X_M + 0.35 and state.pose.distance_to(far_entry) > 0.30:
            return far_entry
        return None

    def _avoid_reef_waypoint(
        self, start_x: float, start_y: float, goal_x: float, goal_y: float
    ) -> tuple[float, float]:
        center_x, center_y = BLUE_REEF_CENTER
        keepout_radius = REEF_OBSTACLE_RADIUS_M + ROBOT_RADIUS_M + REEF_CLEARANCE_M + 0.35
        hard_keepout_radius = REEF_OBSTACLE_RADIUS_M + ROBOT_RADIUS_M + REEF_CLEARANCE_M
        robot_angle = math.atan2(start_y - center_y, start_x - center_x)
        robot_distance = math.hypot(start_x - center_x, start_y - center_y)
        if robot_distance < hard_keepout_radius + 0.05:
            repulsion = max(0.0, hard_keepout_radius + 0.05 - robot_distance)
            return (
                goal_x + 2.0 * repulsion * math.cos(robot_angle),
                goal_y + 2.0 * repulsion * math.sin(robot_angle),
            )
        if math.hypot(goal_x - start_x, goal_y - start_y) < 1.25:
            return goal_x, goal_y
        if not _segment_intersects_circle(
            start_x,
            start_y,
            goal_x,
            goal_y,
            center_x,
            center_y,
            keepout_radius,
        ):
            return goal_x, goal_y

        candidates = []
        for index in range(16):
            angle = (2.0 * math.pi * index) / 16.0
            x = center_x + keepout_radius * math.cos(angle)
            y = center_y + keepout_radius * math.sin(angle)
            if not _segment_intersects_circle(
                start_x,
                start_y,
                x,
                y,
                center_x,
                center_y,
                keepout_radius - 0.05,
            ):
                candidates.append((x, y))

        if not candidates:
            angle = math.atan2(start_y - center_y, start_x - center_x)
            return (
                center_x + keepout_radius * math.cos(angle),
                center_y + keepout_radius * math.sin(angle),
            )

        connected_candidates = [
            point
            for point in candidates
            if not _segment_intersects_circle(
                point[0],
                point[1],
                goal_x,
                goal_y,
                center_x,
                center_y,
                keepout_radius - 0.05,
            )
        ]
        route_candidates = connected_candidates or candidates
        return min(
            route_candidates,
            key=lambda point: (
                math.hypot(point[0] - start_x, point[1] - start_y)
                + math.hypot(point[0] - goal_x, point[1] - goal_y)
            ),
        )

def _segment_intersects_circle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    abx = bx - ax
    aby = by - ay
    length_sq = abx * abx + aby * aby
    if length_sq <= 1e-9:
        return math.hypot(ax - cx, ay - cy) < radius
    t = ((cx - ax) * abx + (cy - ay) * aby) / length_sq
    t = clamp(t, 0.0, 1.0)
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.hypot(closest_x - cx, closest_y - cy) < radius


def _with_min_command(value: float, minimum: float) -> float:
    if abs(value) < 1e-9:
        return 0.0
    if abs(value) >= minimum:
        return value
    return math.copysign(minimum, value)


def _is_in_trench_corridor(y: float) -> bool:
    return y <= TRENCH_CORRIDOR_WIDTH_M or y >= FIELD_WIDTH_M - TRENCH_CORRIDOR_WIDTH_M

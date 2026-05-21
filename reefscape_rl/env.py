"""Simplified REBUILT environment with FUEL pickup and shot physics.

Actions are normalized floats:

    [vx, vy, omega, intake, score]

where vx/vy are field-relative translation commands, omega is a rotation
command, and intake/score are thresholded at > 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

from reefscape_rl.constants import (
    ALLIANCE_SHIFT_DURATION_S,
    AUTO_DURATION_S,
    BLUE_ALLIANCE_ZONE_DEPTH_M,
    BLUE_FUEL_SOURCES,
    BLUE_HUB_CENTER,
    BLUE_MIDFIELD_SWEEP_POINTS,
    BLUE_TRENCH_LINE_X_M,
    CONTROL_PERIOD_S,
    DEFAULT_MAX_FUEL_SCORED,
    DEFAULT_MIDFIELD_FUEL_COUNT,
    DEFAULT_ROBOT_FUEL_CAPACITY,
    DEFAULT_TARGET_LEVEL,
    ENDGAME_START_S,
    FIELD_DIAGONAL_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    FUEL_POINTS_AUTO,
    FUEL_POINTS_TELEOP,
    FUEL_HELD_HEIGHT_M,
    FUEL_HUB_CAPTURE_HEIGHT_TOLERANCE_M,
    FUEL_HUB_TARGET_HEIGHT_M,
    FUEL_RETURN_SCATTER_RADIUS_M,
    FUEL_SHOT_GRAVITY_MPS2,
    FUEL_SHOT_CAPTURE_RADIUS_M,
    FUEL_SHOT_LAUNCH_HEIGHT_M,
    FUEL_SHOT_MAX_AGE_S,
    FUEL_SHOT_SPEED_MPS,
    FUEL_SOURCE_HEIGHT_M,
    HUB_CLEARANCE_M,
    HUB_OBSTACLE_RADIUS_M,
    HUB_SCORING_RADIUS_M,
    INTAKE_DURATION_S,
    INTAKE_RADIUS_M,
    MATCH_DURATION_S,
    MAX_ANGULAR_ACCEL_RADPS2,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_ACCEL_MPS2,
    MAX_LINEAR_SPEED_MPS,
    MECHANISM_ANGULAR_SETTLE_RADPS,
    MECHANISM_LINEAR_SETTLE_MPS,
    ROBOT_RADIUS_M,
    SCORE_DURATION_S,
    TRENCH_CORRIDOR_WIDTH_M,
    TRANSITION_SHIFT_DURATION_S,
)
from reefscape_rl.geometry import Pose2d, Pose3d, angle_to, approach, clamp, normalize_angle


OBSERVATION_FIELDS = (
    "robot_x_norm",
    "robot_y_norm",
    "robot_heading_cos",
    "robot_heading_sin",
    "robot_vx_norm",
    "robot_vy_norm",
    "robot_omega_norm",
    "held_fuel_norm",
    "fuel_source_x_norm",
    "fuel_source_y_norm",
    "goal_x_norm",
    "goal_y_norm",
    "objective_dx_norm",
    "objective_dy_norm",
    "objective_distance_norm",
    "time_remaining_norm",
    "scored_fuel_norm",
    "hub_active",
    "match_phase_norm",
    "active_shots_norm",
)


@dataclass(slots=True)
class FuelShot:
    x: float
    y: float
    z: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    age_s: float = 0.0

    def pose(self) -> Pose2d:
        return Pose2d(self.x, self.y, math.atan2(self.vy_mps, self.vx_mps))

    def pose3d(self) -> Pose3d:
        yaw = math.atan2(self.vy_mps, self.vx_mps)
        horizontal_speed = math.hypot(self.vx_mps, self.vy_mps)
        pitch = math.atan2(self.vz_mps, max(1e-6, horizontal_speed))
        return Pose3d(self.x, self.y, max(0.0, self.z), 0.0, pitch, yaw)


@dataclass(slots=True)
class FuelBall:
    x: float
    y: float
    collected: bool = False

    def pose(self) -> Pose2d:
        return Pose2d(self.x, self.y, 0.0)

    def pose3d(self) -> Pose3d:
        return Pose3d(self.x, self.y, FUEL_SOURCE_HEIGHT_M)


@dataclass(slots=True)
class FallingFuel:
    x: float
    y: float
    z: float
    vx_mps: float
    vy_mps: float
    vz_mps: float

    def pose3d(self) -> Pose3d:
        yaw = math.atan2(self.vy_mps, self.vx_mps)
        horizontal_speed = math.hypot(self.vx_mps, self.vy_mps)
        pitch = math.atan2(self.vz_mps, max(1e-6, horizontal_speed))
        return Pose3d(self.x, self.y, max(0.0, self.z), 0.0, pitch, yaw)


@dataclass(slots=True)
class ReefscapeEnvConfig:
    dt_s: float = CONTROL_PERIOD_S
    episode_duration_s: float = MATCH_DURATION_S
    target_level: str = DEFAULT_TARGET_LEVEL
    max_fuel_scored: int = DEFAULT_MAX_FUEL_SCORED
    max_coral_scored: int | None = None
    midfield_fuel_count: int = DEFAULT_MIDFIELD_FUEL_COUNT
    robot_fuel_capacity: int = DEFAULT_ROBOT_FUEL_CAPACITY
    blue_auto_won: bool = False
    randomize_start: bool = True
    start_pose: Pose2d = field(default_factory=lambda: Pose2d(1.35, FIELD_WIDTH_M / 2.0, 0.0))
    progress_reward_scale: float = 0.35
    timestep_penalty: float = -0.01
    acquire_reward: float = 1.0
    launch_reward: float = 0.12
    inactive_score_penalty: float = -0.4
    missed_shot_penalty: float = -0.15
    invalid_action_penalty: float = -0.03
    boundary_penalty: float = -0.4
    hub_collision_penalty: float = -0.8
    hold_action_reward: float = 0.03
    intake_duration_s: float = INTAKE_DURATION_S
    shot_period_s: float = SCORE_DURATION_S
    auto_mechanisms: bool = False
    freeze_speed_threshold_mps: float = 0.08
    freeze_objective_distance_m: float = 0.45
    freeze_grace_s: float = 0.6
    freeze_penalty_per_s: float = -3.0
    smoothness_reward_scale: float = 0.025
    jerk_penalty_scale: float = -0.035

    def __post_init__(self) -> None:
        if self.max_coral_scored is not None:
            self.max_fuel_scored = self.max_coral_scored
        self.robot_fuel_capacity = max(1, int(self.robot_fuel_capacity))


@dataclass(slots=True)
class ReefscapeState:
    time_s: float
    pose: Pose2d
    vx_mps: float
    vy_mps: float
    omega_radps: float
    held_fuel: int
    scored_fuel: int
    inactive_scored_fuel: int
    missed_fuel: int
    total_reward: float
    current_source_index: int
    current_goal_index: int
    fuel_balls: list[FuelBall]
    fuel_shots: list[FuelShot]
    falling_fuel: list[FallingFuel]
    current_sweep_index: int
    last_objective_distance: float | None
    intake_progress_s: float
    score_progress_s: float
    shot_cooldown_s: float
    is_intaking: bool
    is_scoring: bool
    last_shot_pose: Pose3d | None
    frozen_time_s: float
    last_action_vx: float
    last_action_vy: float
    last_action_omega: float
    smoothness_reward: float

    @property
    def has_fuel(self) -> bool:
        return self.held_fuel > 0

    @has_fuel.setter
    def has_fuel(self, value: bool) -> None:
        self.held_fuel = max(1, self.held_fuel) if value else 0

    @property
    def has_coral(self) -> bool:
        return self.has_fuel

    @has_coral.setter
    def has_coral(self, value: bool) -> None:
        self.has_fuel = value

    @property
    def scored_coral(self) -> int:
        return self.scored_fuel

    @scored_coral.setter
    def scored_coral(self, value: int) -> None:
        self.scored_fuel = int(value)


class ReefscapeEnv:
    """Fuel-cycle environment for the blue alliance side of REBUILT."""

    observation_fields = OBSERVATION_FIELDS
    action_fields = ("vx_norm", "vy_norm", "omega_norm", "intake", "score")

    def __init__(self, config: ReefscapeEnvConfig | None = None):
        self.config = config or ReefscapeEnvConfig()
        self._rng = random.Random()
        self.goal_poses = self._build_goal_poses()
        self.state: ReefscapeState | None = None

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)

        pose = self._initial_pose()
        fuel_balls = self._build_midfield_fuel()
        source_index = self._nearest_source_index(pose, fuel_balls)
        goal_index = self._rng.randrange(len(self.goal_poses))
        self.state = ReefscapeState(
            time_s=0.0,
            pose=pose,
            vx_mps=0.0,
            vy_mps=0.0,
            omega_radps=0.0,
            held_fuel=0,
            scored_fuel=0,
            inactive_scored_fuel=0,
            missed_fuel=0,
            total_reward=0.0,
            current_source_index=source_index,
            current_goal_index=goal_index,
            fuel_balls=fuel_balls,
            fuel_shots=[],
            falling_fuel=[],
            current_sweep_index=0,
            last_objective_distance=None,
            intake_progress_s=0.0,
            score_progress_s=0.0,
            shot_cooldown_s=0.0,
            is_intaking=False,
            is_scoring=False,
            last_shot_pose=None,
            frozen_time_s=0.0,
            last_action_vx=0.0,
            last_action_vy=0.0,
            last_action_omega=0.0,
            smoothness_reward=0.0,
        )
        self.state.last_objective_distance = self._objective_distance()
        return self._observation(), self._info(event_code=0)

    def step(
        self, action: Sequence[float]
    ) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if len(action) != 5:
            raise ValueError(f"Expected 5 action values, got {len(action)}")

        action_vx = clamp(float(action[0]), -1.0, 1.0)
        action_vy = clamp(float(action[1]), -1.0, 1.0)
        action_omega = clamp(float(action[2]), -1.0, 1.0)
        wants_intake = float(action[3]) > 0.5
        wants_score = float(action[4]) > 0.5
        wants_score = wants_score or (state.held_fuel > 0 and self._can_score())
        if self.config.auto_mechanisms:
            wants_score = wants_score or (state.held_fuel > 0 and self._can_score())

        reward = self.config.timestep_penalty
        event_code = 0
        scored_points = 0
        state.is_intaking = False
        state.is_scoring = False
        state.shot_cooldown_s = max(0.0, state.shot_cooldown_s - self.config.dt_s)

        prev_objective_distance = self._objective_distance()
        mechanism_active = False

        if state.held_fuel > 0 and wants_score and self._can_score():
            self._hold_still()
            state.is_scoring = True
            state.intake_progress_s = 0.0
            state.score_progress_s = max(0.0, self.config.shot_period_s - state.shot_cooldown_s)
            reward += self.config.hold_action_reward
            event_code = 4
            mechanism_active = True
            if state.shot_cooldown_s <= 1e-9:
                state.held_fuel -= 1
                state.score_progress_s = self.config.shot_period_s
                state.shot_cooldown_s = self.config.shot_period_s
                self._launch_fuel()
                reward += self.config.launch_reward
        else:
            if wants_score:
                reward += self.config.invalid_action_penalty

        if not mechanism_active:
            self._integrate(action_vx, action_vy, action_omega)
            reward += self._clamp_to_field()
            reward += self._keep_out_of_hub()

        acquired_fuel = self._collect_fuel_under_robot()
        if acquired_fuel:
            reward += self.config.acquire_reward * acquired_fuel
            state.is_intaking = True
            if event_code == 0:
                event_code = 1

        shot_reward, shot_event_code, scored_points = self._advance_fuel_shots()
        self._advance_falling_fuel()
        reward += shot_reward
        if shot_event_code:
            event_code = shot_event_code

        objective_distance = self._objective_distance()
        reward += self.config.progress_reward_scale * (
            prev_objective_distance - objective_distance
        )
        reward += self._settle_reward(objective_distance)
        reward += self._smoothness_reward(
            action_vx,
            action_vy,
            action_omega,
            objective_distance,
            mechanism_active,
        )
        reward += self._freeze_penalty(objective_distance, mechanism_active)

        if not mechanism_active:
            state.intake_progress_s = 0.0
            state.score_progress_s = 0.0

        state.time_s += self.config.dt_s
        state.total_reward += reward
        state.last_objective_distance = self._objective_distance()

        terminated = state.scored_fuel >= self.config.max_fuel_scored
        truncated = state.time_s >= self.config.episode_duration_s - 1e-9
        info = self._info(event_code=event_code, scored_points=scored_points)
        return self._observation(), reward, terminated, truncated, info

    def is_blue_hub_active(self) -> bool:
        state = self._require_state()
        phase = self._match_phase_code(state.time_s)
        if phase in {0, 1, 6}:
            return True
        shift_index = phase - 2
        return shift_index % 2 == (1 if self.config.blue_auto_won else 0)

    def current_objective_pose(self) -> Pose2d:
        state = self._require_state()
        if state.held_fuel > 0 and self.is_blue_hub_active():
            return self.current_goal_pose()
        ball = self._nearest_available_fuel_ball(state.pose)
        if ball is not None:
            return ball.pose()
        sweep = BLUE_MIDFIELD_SWEEP_POINTS[state.current_sweep_index % len(BLUE_MIDFIELD_SWEEP_POINTS)]
        return Pose2d(sweep[0], sweep[1], 0.0)

    def current_goal_pose(self) -> Pose2d:
        state = self._require_state()
        return self.goal_poses[state.current_goal_index]

    def current_fuel_pose(self) -> Pose2d:
        state = self._require_state()
        if state.fuel_shots:
            return state.fuel_shots[0].pose()
        if state.held_fuel > 0:
            return Pose2d(state.pose.x, state.pose.y, state.pose.heading)
        ball = self._nearest_available_fuel_ball(state.pose)
        if ball is not None:
            return ball.pose()
        sweep = BLUE_MIDFIELD_SWEEP_POINTS[state.current_sweep_index % len(BLUE_MIDFIELD_SWEEP_POINTS)]
        return Pose2d(sweep[0], sweep[1], 0.0)

    def current_fuel_pose3d(self) -> Pose3d:
        state = self._require_state()
        if state.fuel_shots:
            return state.fuel_shots[0].pose3d()
        if state.held_fuel > 0:
            return Pose3d(state.pose.x, state.pose.y, FUEL_HELD_HEIGHT_M, 0.0, 0.0, state.pose.heading)
        ball = self._nearest_available_fuel_ball(state.pose)
        if ball is not None:
            return ball.pose3d()
        sweep = BLUE_MIDFIELD_SWEEP_POINTS[state.current_sweep_index % len(BLUE_MIDFIELD_SWEEP_POINTS)]
        return Pose3d(sweep[0], sweep[1], FUEL_SOURCE_HEIGHT_M)

    def fuel_poses3d(self) -> list[Pose3d]:
        state = self._require_state()
        poses = [ball.pose3d() for ball in state.fuel_balls if not ball.collected]
        poses.extend(shot.pose3d() for shot in state.fuel_shots)
        poses.extend(fuel.pose3d() for fuel in state.falling_fuel)
        return poses

    def current_coral_pose(self) -> Pose2d:
        return self.current_fuel_pose()

    def _initial_pose(self) -> Pose2d:
        if not self.config.randomize_start:
            return Pose2d(
                self.config.start_pose.x,
                self.config.start_pose.y,
                self.config.start_pose.heading,
            )
        return Pose2d(
            x=self.config.start_pose.x + self._rng.uniform(-0.35, 0.35),
            y=clamp(
                self.config.start_pose.y + self._rng.uniform(-1.25, 1.25),
                ROBOT_RADIUS_M,
                FIELD_WIDTH_M - ROBOT_RADIUS_M,
            ),
            heading=self._rng.uniform(-math.pi, math.pi),
        )

    def _integrate(self, vx_norm: float, vy_norm: float, omega_norm: float) -> None:
        state = self._require_state()
        dt = self.config.dt_s
        target_vx = vx_norm * MAX_LINEAR_SPEED_MPS
        target_vy = vy_norm * MAX_LINEAR_SPEED_MPS
        target_omega = omega_norm * MAX_ANGULAR_SPEED_RADPS
        old_x = state.pose.x
        old_y = state.pose.y

        state.vx_mps = approach(state.vx_mps, target_vx, MAX_LINEAR_ACCEL_MPS2 * dt)
        state.vy_mps = approach(state.vy_mps, target_vy, MAX_LINEAR_ACCEL_MPS2 * dt)
        state.omega_radps = approach(
            state.omega_radps, target_omega, MAX_ANGULAR_ACCEL_RADPS2 * dt
        )

        state.pose.x += state.vx_mps * dt
        state.pose.y += state.vy_mps * dt
        state.pose.heading = normalize_angle(state.pose.heading + state.omega_radps * dt)
        self._enforce_trench_crossing(old_x, old_y)

    def _clamp_to_field(self) -> float:
        state = self._require_state()
        old_x = state.pose.x
        old_y = state.pose.y
        state.pose.x = clamp(state.pose.x, ROBOT_RADIUS_M, FIELD_LENGTH_M - ROBOT_RADIUS_M)
        state.pose.y = clamp(state.pose.y, ROBOT_RADIUS_M, FIELD_WIDTH_M - ROBOT_RADIUS_M)
        if state.pose.x != old_x:
            state.vx_mps = 0.0
        if state.pose.y != old_y:
            state.vy_mps = 0.0
        return self.config.boundary_penalty if state.pose.x != old_x or state.pose.y != old_y else 0.0

    def _enforce_trench_crossing(self, old_x: float, old_y: float) -> None:
        state = self._require_state()
        crossed_trench_line = (
            (old_x - BLUE_TRENCH_LINE_X_M) * (state.pose.x - BLUE_TRENCH_LINE_X_M) <= 0.0
            and abs(state.pose.x - old_x) > 1e-9
        )
        if not crossed_trench_line:
            return

        if self._is_in_trench_corridor((old_y + state.pose.y) * 0.5):
            return

        state.pose.x = (
            BLUE_TRENCH_LINE_X_M - ROBOT_RADIUS_M
            if old_x < BLUE_TRENCH_LINE_X_M
            else BLUE_TRENCH_LINE_X_M + ROBOT_RADIUS_M
        )
        state.vx_mps = 0.0

    def _is_in_trench_corridor(self, y: float) -> bool:
        return y <= TRENCH_CORRIDOR_WIDTH_M or y >= FIELD_WIDTH_M - TRENCH_CORRIDOR_WIDTH_M

    def _observation(self) -> list[float]:
        state = self._require_state()
        fuel_pose = self.current_fuel_pose()
        goal_pose = self.current_goal_pose()
        objective = self.current_objective_pose()
        dx = objective.x - state.pose.x
        dy = objective.y - state.pose.y
        return [
            state.pose.x / FIELD_LENGTH_M,
            state.pose.y / FIELD_WIDTH_M,
            math.cos(state.pose.heading),
            math.sin(state.pose.heading),
            state.vx_mps / MAX_LINEAR_SPEED_MPS,
            state.vy_mps / MAX_LINEAR_SPEED_MPS,
            state.omega_radps / MAX_ANGULAR_SPEED_RADPS,
            state.held_fuel / self.config.robot_fuel_capacity,
            fuel_pose.x / FIELD_LENGTH_M,
            fuel_pose.y / FIELD_WIDTH_M,
            goal_pose.x / FIELD_LENGTH_M,
            goal_pose.y / FIELD_WIDTH_M,
            dx / FIELD_LENGTH_M,
            dy / FIELD_WIDTH_M,
            math.hypot(dx, dy) / FIELD_DIAGONAL_M,
            max(0.0, self.config.episode_duration_s - state.time_s)
            / self.config.episode_duration_s,
            state.scored_fuel / max(1, self.config.max_fuel_scored),
            1.0 if self.is_blue_hub_active() else 0.0,
            self._match_phase_code(state.time_s) / 6.0,
            min(1.0, len(state.fuel_shots) / self.config.robot_fuel_capacity),
        ]

    def _info(self, *, event_code: int, scored_points: int = 0) -> dict[str, Any]:
        state = self._require_state()
        fuel_pose = self.current_fuel_pose()
        goal_pose = self.current_goal_pose()
        phase_code = self._match_phase_code(state.time_s)
        return {
            "time_s": state.time_s,
            "event_code": event_code,
            "scored_points": scored_points,
            "scored_fuel": state.scored_fuel,
            "inactive_scored_fuel": state.inactive_scored_fuel,
            "missed_fuel": state.missed_fuel,
            "held_fuel": state.held_fuel,
            "has_fuel": state.has_fuel,
            "active_shots": len(state.fuel_shots),
            "falling_fuel": len(state.falling_fuel),
            "hub_active": self.is_blue_hub_active(),
            "match_phase": self._match_phase_name(phase_code),
            "match_phase_code": phase_code,
            "total_reward": state.total_reward,
            "robot_pose": state.pose,
            "fuel_pose": fuel_pose,
            "fuel_pose3d": self.current_fuel_pose3d(),
            "goal_pose": goal_pose,
            "last_shot_pose": state.last_shot_pose,
            "target_level": self.config.target_level,
            "objective_distance_m": self._objective_distance(),
            "intake_progress_s": state.intake_progress_s,
            "score_progress_s": state.score_progress_s,
            "is_intaking": state.is_intaking,
            "is_scoring": state.is_scoring,
            "intake_duration_s": self.config.intake_duration_s,
            "score_duration_s": self.config.shot_period_s,
            "shot_period_s": self.config.shot_period_s,
            "frozen_time_s": state.frozen_time_s,
            "smoothness_reward": state.smoothness_reward,
            "source_remaining": self._current_source_remaining(),
            "scored_coral": state.scored_fuel,
            "has_coral": state.has_fuel,
            "coral_pose": fuel_pose,
        }

    def _build_goal_poses(self) -> list[Pose2d]:
        center_x, center_y = BLUE_HUB_CENTER
        poses: list[Pose2d] = []
        for degrees in (150.0, 165.0, 180.0, 195.0, 210.0):
            angle = math.radians(degrees)
            x = center_x + HUB_SCORING_RADIUS_M * math.cos(angle)
            y = center_y + HUB_SCORING_RADIUS_M * math.sin(angle)
            x = clamp(x, ROBOT_RADIUS_M, BLUE_ALLIANCE_ZONE_DEPTH_M - 0.15)
            y = clamp(y, ROBOT_RADIUS_M, FIELD_WIDTH_M - ROBOT_RADIUS_M)
            poses.append(Pose2d(x, y, angle_to(x, y, center_x, center_y)))
        return poses

    def _next_goal_index(self) -> int:
        state = self._require_state()
        return (state.current_goal_index + 1) % len(self.goal_poses)

    def _build_midfield_fuel(self) -> list[FuelBall]:
        count = max(0, int(self.config.midfield_fuel_count))
        if count <= 0:
            return []

        columns = 10
        rows = math.ceil(count / columns)
        x_min = FIELD_LENGTH_M / 2.0 - 1.05
        x_spacing = 2.10 / max(1, columns - 1)
        y_min = 1.45
        y_spacing = (FIELD_WIDTH_M - 2.90) / max(1, rows - 1)
        balls: list[FuelBall] = []
        for index in range(count):
            col = index % columns
            row = index // columns
            stagger = 0.10 if row % 2 else 0.0
            x = x_min + col * x_spacing + stagger
            y = y_min + row * y_spacing
            balls.append(FuelBall(x, y))
        return balls

    def _nearest_source_index(
        self, pose: Pose2d, fuel_balls: list[FuelBall] | None = None
    ) -> int:
        balls = fuel_balls if fuel_balls is not None else self._require_state().fuel_balls
        available = [index for index, ball in enumerate(balls) if not ball.collected]
        if not available:
            return 0
        return min(available, key=lambda index: pose.distance_to((balls[index].x, balls[index].y)))

    def _nearest_available_fuel_ball(self, pose: Pose2d) -> FuelBall | None:
        state = self._require_state()
        available = [ball for ball in state.fuel_balls if not ball.collected]
        if not available:
            return None
        return min(available, key=lambda ball: pose.distance_to((ball.x, ball.y)))

    def _current_source_remaining(self) -> int:
        state = self._require_state()
        return sum(1 for ball in state.fuel_balls if not ball.collected)

    def _collect_fuel_under_robot(self) -> int:
        state = self._require_state()
        if state.held_fuel >= self.config.robot_fuel_capacity:
            return 0

        acquired = 0
        for ball in state.fuel_balls:
            if ball.collected:
                continue
            if state.pose.distance_to((ball.x, ball.y)) > INTAKE_RADIUS_M:
                continue
            ball.collected = True
            state.held_fuel += 1
            acquired += 1
            if state.held_fuel >= self.config.robot_fuel_capacity:
                break

        if acquired:
            state.current_source_index = self._nearest_source_index(state.pose)
            state.intake_progress_s = 0.0
        return acquired

    def _objective_distance(self) -> float:
        state = self._require_state()
        return state.pose.distance_to(self.current_objective_pose())

    def _can_score(self) -> bool:
        state = self._require_state()
        return self.is_blue_hub_active() and state.pose.x >= BLUE_TRENCH_LINE_X_M

    def _launch_fuel(self) -> None:
        state = self._require_state()
        dx = BLUE_HUB_CENTER[0] - state.pose.x
        dy = BLUE_HUB_CENTER[1] - state.pose.y
        distance = max(1e-6, math.hypot(dx, dy))
        ux = dx / distance
        uy = dy / distance
        shot = FuelShot(
            x=state.pose.x + ux * ROBOT_RADIUS_M,
            y=state.pose.y + uy * ROBOT_RADIUS_M,
            z=FUEL_SHOT_LAUNCH_HEIGHT_M,
            vx_mps=ux * FUEL_SHOT_SPEED_MPS + 0.25 * state.vx_mps,
            vy_mps=uy * FUEL_SHOT_SPEED_MPS + 0.25 * state.vy_mps,
            vz_mps=self._shot_vertical_speed(distance),
        )
        state.fuel_shots.append(shot)
        state.last_shot_pose = shot.pose3d()

    def _advance_fuel_shots(self) -> tuple[float, int, int]:
        state = self._require_state()
        reward = 0.0
        event_code = 0
        scored_points = 0
        remaining: list[FuelShot] = []
        for shot in state.fuel_shots:
            shot.x += shot.vx_mps * self.config.dt_s
            shot.y += shot.vy_mps * self.config.dt_s
            shot.z += shot.vz_mps * self.config.dt_s
            shot.vz_mps -= FUEL_SHOT_GRAVITY_MPS2 * self.config.dt_s
            shot.age_s += self.config.dt_s
            state.last_shot_pose = shot.pose3d()

            shot_at_hub = math.hypot(shot.x - BLUE_HUB_CENTER[0], shot.y - BLUE_HUB_CENTER[1]) <= FUEL_SHOT_CAPTURE_RADIUS_M
            shot_at_height = abs(shot.z - FUEL_HUB_TARGET_HEIGHT_M) <= FUEL_HUB_CAPTURE_HEIGHT_TOLERANCE_M
            if shot_at_hub and shot_at_height:
                if self.is_blue_hub_active():
                    state.scored_fuel += 1
                    points = self._fuel_point_value()
                    scored_points += points
                    reward += float(points)
                    event_code = 2
                    self._spawn_falling_fuel()
                else:
                    state.inactive_scored_fuel += 1
                    reward += self.config.inactive_score_penalty
                    event_code = 7
                state.current_source_index = self._nearest_source_index(state.pose)
                state.current_goal_index = self._next_goal_index()
                continue

            out_of_field = (
                shot.x < 0.0
                or shot.x > FIELD_LENGTH_M
                or shot.y < 0.0
                or shot.y > FIELD_WIDTH_M
            )
            hit_floor = shot.z <= 0.0 and shot.age_s > 0.05
            if shot.age_s >= FUEL_SHOT_MAX_AGE_S or out_of_field or hit_floor:
                state.missed_fuel += 1
                reward += self.config.missed_shot_penalty
                event_code = event_code or 8
                continue

            remaining.append(shot)

        state.fuel_shots = remaining
        return reward, event_code, scored_points

    def _spawn_falling_fuel(self) -> None:
        state = self._require_state()
        angle = self._rng.uniform(-math.pi, math.pi)
        speed = self._rng.uniform(0.2, 0.8)
        state.falling_fuel.append(
            FallingFuel(
                x=BLUE_HUB_CENTER[0],
                y=BLUE_HUB_CENTER[1],
                z=FUEL_HUB_TARGET_HEIGHT_M,
                vx_mps=math.cos(angle) * speed,
                vy_mps=math.sin(angle) * speed,
                vz_mps=0.0,
            )
        )

    def _advance_falling_fuel(self) -> None:
        state = self._require_state()
        remaining: list[FallingFuel] = []
        for fuel in state.falling_fuel:
            fuel.x += fuel.vx_mps * self.config.dt_s
            fuel.y += fuel.vy_mps * self.config.dt_s
            fuel.z += fuel.vz_mps * self.config.dt_s
            fuel.vz_mps -= FUEL_SHOT_GRAVITY_MPS2 * self.config.dt_s
            if fuel.z > FUEL_SOURCE_HEIGHT_M:
                remaining.append(fuel)
                continue

            dx = fuel.x - BLUE_HUB_CENTER[0]
            dy = fuel.y - BLUE_HUB_CENTER[1]
            distance = max(1e-6, math.hypot(dx, dy))
            landing_radius = max(FUEL_RETURN_SCATTER_RADIUS_M, distance)
            x = BLUE_HUB_CENTER[0] + (dx / distance) * landing_radius
            y = BLUE_HUB_CENTER[1] + (dy / distance) * landing_radius
            state.fuel_balls.append(
                FuelBall(
                    clamp(x, ROBOT_RADIUS_M, FIELD_LENGTH_M - ROBOT_RADIUS_M),
                    clamp(y, ROBOT_RADIUS_M, FIELD_WIDTH_M - ROBOT_RADIUS_M),
                )
            )
        state.falling_fuel = remaining

    def _shot_vertical_speed(self, horizontal_distance_m: float) -> float:
        flight_time_s = max(0.25, horizontal_distance_m / max(1e-6, FUEL_SHOT_SPEED_MPS))
        return (
            FUEL_HUB_TARGET_HEIGHT_M
            - FUEL_SHOT_LAUNCH_HEIGHT_M
            + 0.5 * FUEL_SHOT_GRAVITY_MPS2 * flight_time_s * flight_time_s
        ) / flight_time_s

    def _fuel_point_value(self) -> int:
        state = self._require_state()
        return FUEL_POINTS_AUTO if state.time_s < AUTO_DURATION_S else FUEL_POINTS_TELEOP

    def _is_settled(self) -> bool:
        state = self._require_state()
        linear_speed = math.hypot(state.vx_mps, state.vy_mps)
        return (
            linear_speed <= MECHANISM_LINEAR_SETTLE_MPS
            and abs(state.omega_radps) <= MECHANISM_ANGULAR_SETTLE_RADPS
        )

    def _settle_reward(self, objective_distance: float) -> float:
        if objective_distance > 0.45:
            return 0.0
        state = self._require_state()
        speed = math.hypot(state.vx_mps, state.vy_mps)
        return 0.05 * (0.45 - objective_distance) - 0.03 * speed

    def _freeze_penalty(self, objective_distance: float, mechanism_active: bool) -> float:
        state = self._require_state()
        speed = math.hypot(state.vx_mps, state.vy_mps)
        is_freezing = (
            not mechanism_active
            and objective_distance >= self.config.freeze_objective_distance_m
            and speed <= self.config.freeze_speed_threshold_mps
            and abs(state.omega_radps) <= 0.25
        )
        if not is_freezing:
            state.frozen_time_s = 0.0
            return 0.0

        state.frozen_time_s += self.config.dt_s
        excess_s = max(0.0, state.frozen_time_s - self.config.freeze_grace_s)
        if excess_s <= 0.0:
            return 0.0
        return self.config.freeze_penalty_per_s * self.config.dt_s * (1.0 + 2.0 * excess_s)

    def _smoothness_reward(
        self,
        action_vx: float,
        action_vy: float,
        action_omega: float,
        objective_distance: float,
        mechanism_active: bool,
    ) -> float:
        state = self._require_state()
        if mechanism_active:
            state.smoothness_reward = 0.0
            state.last_action_vx = 0.0
            state.last_action_vy = 0.0
            state.last_action_omega = 0.0
            return 0.0

        delta_vx = action_vx - state.last_action_vx
        delta_vy = action_vy - state.last_action_vy
        delta_omega = action_omega - state.last_action_omega
        jerk = math.sqrt(delta_vx * delta_vx + delta_vy * delta_vy + 0.35 * delta_omega * delta_omega)
        speed = math.hypot(state.vx_mps, state.vy_mps)
        moving_toward_objective = (
            speed > self.config.freeze_speed_threshold_mps
            and objective_distance > 0.45
        )
        smooth_bonus = self.config.smoothness_reward_scale / (1.0 + 3.0 * jerk) if moving_toward_objective else 0.0
        reward = smooth_bonus + self.config.jerk_penalty_scale * jerk
        state.smoothness_reward = reward
        state.last_action_vx = action_vx
        state.last_action_vy = action_vy
        state.last_action_omega = action_omega
        return reward

    def _hold_still(self) -> None:
        state = self._require_state()
        state.vx_mps = 0.0
        state.vy_mps = 0.0
        state.omega_radps = 0.0

    def _keep_out_of_hub(self) -> float:
        state = self._require_state()
        center_x, center_y = BLUE_HUB_CENTER
        dx = state.pose.x - center_x
        dy = state.pose.y - center_y
        distance = math.hypot(dx, dy)
        min_distance = HUB_OBSTACLE_RADIUS_M + ROBOT_RADIUS_M + HUB_CLEARANCE_M
        if distance >= min_distance:
            return 0.0

        if distance < 1e-6:
            dx = -1.0
            dy = 0.0
            distance = 1.0

        scale = min_distance / distance
        state.pose.x = center_x + dx * scale
        state.pose.y = center_y + dy * scale
        state.vx_mps = 0.0
        state.vy_mps = 0.0
        return self.config.hub_collision_penalty

    def _match_phase_code(self, time_s: float) -> int:
        if time_s < AUTO_DURATION_S:
            return 0
        transition_end = AUTO_DURATION_S + TRANSITION_SHIFT_DURATION_S
        if time_s < transition_end:
            return 1
        if time_s >= ENDGAME_START_S:
            return 6
        shift_index = int((time_s - transition_end) // ALLIANCE_SHIFT_DURATION_S)
        return int(clamp(2 + shift_index, 2, 5))

    def _match_phase_name(self, phase_code: int) -> str:
        return (
            "auto",
            "transition",
            "shift_1",
            "shift_2",
            "shift_3",
            "shift_4",
            "endgame",
        )[int(phase_code)]

    def _require_state(self) -> ReefscapeState:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        return self.state

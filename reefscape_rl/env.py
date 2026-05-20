"""Simplified REEFSCAPE environment.

The API mirrors Gymnasium's reset/step shape while staying dependency-free:

    observation, info = env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(action)

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
    BLUE_CORAL_STATIONS,
    BLUE_REEF_CENTER,
    CONTROL_PERIOD_S,
    DEFAULT_MAX_CORAL_SCORED,
    DEFAULT_TARGET_LEVEL,
    FIELD_DIAGONAL_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    INTAKE_RADIUS_M,
    INTAKE_DURATION_S,
    MATCH_DURATION_S,
    MAX_ANGULAR_ACCEL_RADPS2,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_LINEAR_ACCEL_MPS2,
    MAX_LINEAR_SPEED_MPS,
    MECHANISM_ANGULAR_SETTLE_RADPS,
    MECHANISM_LINEAR_SETTLE_MPS,
    REEF_CLEARANCE_M,
    REEF_OBSTACLE_RADIUS_M,
    REEF_SCORING_RADIUS_M,
    ROBOT_RADIUS_M,
    SCORE_HEADING_TOLERANCE_RAD,
    SCORE_DURATION_S,
    SCORE_RADIUS_M,
    SCORE_TRIGGER_RADIUS_M,
    SCORING_POINTS_TELEOP,
)
from reefscape_rl.geometry import Pose2d, angle_to, approach, clamp, normalize_angle


OBSERVATION_FIELDS = (
    "robot_x_norm",
    "robot_y_norm",
    "robot_heading_cos",
    "robot_heading_sin",
    "robot_vx_norm",
    "robot_vy_norm",
    "robot_omega_norm",
    "has_coral",
    "coral_x_norm",
    "coral_y_norm",
    "goal_x_norm",
    "goal_y_norm",
    "objective_dx_norm",
    "objective_dy_norm",
    "objective_distance_norm",
    "time_remaining_norm",
    "scored_coral_norm",
)


@dataclass(slots=True)
class ReefscapeEnvConfig:
    dt_s: float = CONTROL_PERIOD_S
    episode_duration_s: float = MATCH_DURATION_S
    target_level: str = DEFAULT_TARGET_LEVEL
    max_coral_scored: int = DEFAULT_MAX_CORAL_SCORED
    randomize_start: bool = True
    start_pose: Pose2d = field(default_factory=lambda: Pose2d(1.35, FIELD_WIDTH_M / 2.0, 0.0))
    progress_reward_scale: float = 0.35
    timestep_penalty: float = -0.01
    acquire_reward: float = 1.0
    invalid_action_penalty: float = -0.03
    boundary_penalty: float = -0.4
    reef_collision_penalty: float = -0.8
    hold_action_reward: float = 0.03
    intake_duration_s: float = INTAKE_DURATION_S
    score_duration_s: float = SCORE_DURATION_S
    auto_mechanisms: bool = False


@dataclass(slots=True)
class ReefscapeState:
    time_s: float
    pose: Pose2d
    vx_mps: float
    vy_mps: float
    omega_radps: float
    has_coral: bool
    scored_coral: int
    total_reward: float
    current_source_index: int
    current_goal_index: int
    last_objective_distance: float | None
    intake_progress_s: float
    score_progress_s: float
    is_intaking: bool
    is_scoring: bool


class ReefscapeEnv:
    """Coral-cycle environment for the blue alliance side of REEFSCAPE."""

    observation_fields = OBSERVATION_FIELDS
    action_fields = ("vx_norm", "vy_norm", "omega_norm", "intake", "score")

    def __init__(self, config: ReefscapeEnvConfig | None = None):
        self.config = config or ReefscapeEnvConfig()
        if self.config.target_level not in SCORING_POINTS_TELEOP:
            raise ValueError(f"Unknown target level: {self.config.target_level}")
        self._rng = random.Random()
        self.goal_poses = self._build_goal_poses()
        self.state: ReefscapeState | None = None

    def reset(self, *, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)

        pose = self._initial_pose()
        source_index = self._nearest_source_index(pose)
        goal_index = self._rng.randrange(len(self.goal_poses))
        self.state = ReefscapeState(
            time_s=0.0,
            pose=pose,
            vx_mps=0.0,
            vy_mps=0.0,
            omega_radps=0.0,
            has_coral=False,
            scored_coral=0,
            total_reward=0.0,
            current_source_index=source_index,
            current_goal_index=goal_index,
            last_objective_distance=None,
            intake_progress_s=0.0,
            score_progress_s=0.0,
            is_intaking=False,
            is_scoring=False,
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
        if self.config.auto_mechanisms:
            wants_intake = wants_intake or (not state.has_coral and self._can_intake())
            wants_score = wants_score or (state.has_coral and self._can_score())

        reward = self.config.timestep_penalty
        event_code = 0
        scored_points = 0
        state.is_intaking = False
        state.is_scoring = False

        prev_objective_distance = self._objective_distance()
        mechanism_active = False

        if not state.has_coral and wants_intake and self._can_intake():
            self._hold_still()
            state.is_intaking = True
            state.score_progress_s = 0.0
            state.intake_progress_s += self.config.dt_s
            reward += self.config.hold_action_reward
            event_code = 3
            mechanism_active = True
            if state.intake_progress_s >= self.config.intake_duration_s:
                state.has_coral = True
                state.intake_progress_s = 0.0
                reward += self.config.acquire_reward
                event_code = 1
        elif state.has_coral and wants_score and self._can_score():
            self._hold_still()
            state.is_scoring = True
            state.intake_progress_s = 0.0
            state.score_progress_s += self.config.dt_s
            reward += self.config.hold_action_reward
            event_code = 4
            mechanism_active = True
            if state.score_progress_s >= self.config.score_duration_s:
                state.has_coral = False
                state.score_progress_s = 0.0
                state.scored_coral += 1
                scored_points = SCORING_POINTS_TELEOP[self.config.target_level]
                reward += float(scored_points)
                event_code = 2
                state.current_source_index = self._nearest_source_index(state.pose)
                state.current_goal_index = self._next_goal_index()
        else:
            if wants_intake:
                reward += self.config.invalid_action_penalty
            if wants_score:
                reward += self.config.invalid_action_penalty

        if not mechanism_active:
            self._integrate(action_vx, action_vy, action_omega)
            reward += self._clamp_to_field()
            reward += self._keep_out_of_reef()

        objective_distance = self._objective_distance()
        reward += self.config.progress_reward_scale * (
            prev_objective_distance - objective_distance
        )
        reward += self._settle_reward(objective_distance)

        if not mechanism_active:
            state.intake_progress_s = 0.0
            state.score_progress_s = 0.0

        state.time_s += self.config.dt_s
        state.total_reward += reward
        state.last_objective_distance = self._objective_distance()

        terminated = state.scored_coral >= self.config.max_coral_scored
        truncated = state.time_s >= self.config.episode_duration_s - 1e-9
        info = self._info(event_code=event_code, scored_points=scored_points)
        return self._observation(), reward, terminated, truncated, info

    def current_objective_pose(self) -> Pose2d:
        state = self._require_state()
        if state.has_coral:
            return self.goal_poses[state.current_goal_index]
        source = BLUE_CORAL_STATIONS[state.current_source_index]
        return Pose2d(source[0], source[1], 0.0)

    def current_goal_pose(self) -> Pose2d:
        state = self._require_state()
        return self.goal_poses[state.current_goal_index]

    def current_coral_pose(self) -> Pose2d:
        state = self._require_state()
        if state.has_coral:
            return Pose2d(state.pose.x, state.pose.y, state.pose.heading)
        source = BLUE_CORAL_STATIONS[state.current_source_index]
        return Pose2d(source[0], source[1], 0.0)

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

        state.vx_mps = approach(state.vx_mps, target_vx, MAX_LINEAR_ACCEL_MPS2 * dt)
        state.vy_mps = approach(state.vy_mps, target_vy, MAX_LINEAR_ACCEL_MPS2 * dt)
        state.omega_radps = approach(
            state.omega_radps, target_omega, MAX_ANGULAR_ACCEL_RADPS2 * dt
        )

        state.pose.x += state.vx_mps * dt
        state.pose.y += state.vy_mps * dt
        state.pose.heading = normalize_angle(state.pose.heading + state.omega_radps * dt)

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

    def _observation(self) -> list[float]:
        state = self._require_state()
        coral_pose = self.current_coral_pose()
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
            1.0 if state.has_coral else 0.0,
            coral_pose.x / FIELD_LENGTH_M,
            coral_pose.y / FIELD_WIDTH_M,
            goal_pose.x / FIELD_LENGTH_M,
            goal_pose.y / FIELD_WIDTH_M,
            dx / FIELD_LENGTH_M,
            dy / FIELD_WIDTH_M,
            math.hypot(dx, dy) / FIELD_DIAGONAL_M,
            max(0.0, self.config.episode_duration_s - state.time_s)
            / self.config.episode_duration_s,
            state.scored_coral / max(1, self.config.max_coral_scored),
        ]

    def _info(self, *, event_code: int, scored_points: int = 0) -> dict[str, Any]:
        state = self._require_state()
        coral_pose = self.current_coral_pose()
        goal_pose = self.current_goal_pose()
        return {
            "time_s": state.time_s,
            "event_code": event_code,
            "scored_points": scored_points,
            "scored_coral": state.scored_coral,
            "has_coral": state.has_coral,
            "total_reward": state.total_reward,
            "robot_pose": state.pose,
            "coral_pose": coral_pose,
            "goal_pose": goal_pose,
            "target_level": self.config.target_level,
            "objective_distance_m": self._objective_distance(),
            "intake_progress_s": state.intake_progress_s,
            "score_progress_s": state.score_progress_s,
            "is_intaking": state.is_intaking,
            "is_scoring": state.is_scoring,
            "intake_duration_s": self.config.intake_duration_s,
            "score_duration_s": self.config.score_duration_s,
        }

    def _build_goal_poses(self) -> list[Pose2d]:
        center_x, center_y = BLUE_REEF_CENTER
        poses: list[Pose2d] = []
        for index in range(12):
            angle = (2.0 * math.pi * index) / 12.0
            x = center_x + REEF_SCORING_RADIUS_M * math.cos(angle)
            y = center_y + REEF_SCORING_RADIUS_M * math.sin(angle)
            heading = angle_to(x, y, center_x, center_y)
            poses.append(Pose2d(x, y, heading))
        return poses

    def _next_goal_index(self) -> int:
        state = self._require_state()
        return (state.current_goal_index + 1) % len(self.goal_poses)

    def _nearest_source_index(self, pose: Pose2d) -> int:
        distances = [pose.distance_to(source) for source in BLUE_CORAL_STATIONS]
        return min(range(len(distances)), key=distances.__getitem__)

    def _distance_to_source(self) -> float:
        state = self._require_state()
        return state.pose.distance_to(BLUE_CORAL_STATIONS[state.current_source_index])

    def _objective_distance(self) -> float:
        state = self._require_state()
        return state.pose.distance_to(self.current_objective_pose())

    def _can_score(self) -> bool:
        state = self._require_state()
        goal = self.current_goal_pose()
        distance_ok = state.pose.distance_to(goal) <= SCORE_TRIGGER_RADIUS_M
        heading_error = abs(normalize_angle(goal.heading - state.pose.heading))
        heading_ok = heading_error <= SCORE_HEADING_TOLERANCE_RAD
        return distance_ok and heading_ok and self._is_settled()

    def _can_intake(self) -> bool:
        return self._distance_to_source() <= INTAKE_RADIUS_M and self._is_settled()

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

    def _hold_still(self) -> None:
        state = self._require_state()
        state.vx_mps = 0.0
        state.vy_mps = 0.0
        state.omega_radps = 0.0

    def _keep_out_of_reef(self) -> float:
        state = self._require_state()
        center_x, center_y = BLUE_REEF_CENTER
        dx = state.pose.x - center_x
        dy = state.pose.y - center_y
        distance = math.hypot(dx, dy)
        min_distance = REEF_OBSTACLE_RADIUS_M + ROBOT_RADIUS_M + REEF_CLEARANCE_M
        if distance >= min_distance:
            return 0.0

        if distance < 1e-6:
            dx = 1.0
            dy = 0.0
            distance = 1.0

        scale = min_distance / distance
        state.pose.x = center_x + dx * scale
        state.pose.y = center_y + dy * scale
        state.vx_mps = 0.0
        state.vy_mps = 0.0
        return self.config.reef_collision_penalty

    def _require_state(self) -> ReefscapeState:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        return self.state

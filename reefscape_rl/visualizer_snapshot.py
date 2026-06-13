"""State snapshots for the REEFSCAPE-specific desktop visualizer."""

from __future__ import annotations

from typing import Any, Sequence

from reefscape_rl.constants import (
    BLUE_CORAL_STATIONS,
    BLUE_REEF_CENTER,
    BLUE_SIDE_OTHER_ROBOT_PATH,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    REEF_OBSTACLE_RADIUS_M,
    REEF_SCORING_RADIUS_M,
    ROBOT_RADIUS_M,
    SCORING_POINTS_TELEOP,
)
from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import Pose2d
from reefscape_rl.mental_visualizer import build_mental_snapshot


def build_visualizer_snapshot(
    env: ReefscapeEnv,
    *,
    action: Sequence[float],
    reward: float,
    policy_name: str,
    running: bool,
) -> dict[str, Any]:
    """Return a JSON-serializable 2025 game snapshot for the desktop visualizer."""

    state = env.state
    if state is None:
        raise RuntimeError("Cannot snapshot before env.reset().")

    info = env._info(event_code=0)  # Internal state format already feeds logs and tests.
    mental = build_mental_snapshot(env, action)
    objective = env.current_objective_pose()
    goal = env.current_goal_pose()
    coral = env.current_coral_pose()
    time_remaining = max(0.0, env.config.episode_duration_s - state.time_s)

    return {
        "running": running,
        "policy": policy_name,
        "field": {
            "lengthM": FIELD_LENGTH_M,
            "widthM": FIELD_WIDTH_M,
            "reefCenter": _point(BLUE_REEF_CENTER),
            "reefObstacleRadiusM": REEF_OBSTACLE_RADIUS_M,
            "reefScoringRadiusM": REEF_SCORING_RADIUS_M,
            "robotRadiusM": ROBOT_RADIUS_M,
            "coralStations": [_point(point) for point in BLUE_CORAL_STATIONS],
            "trafficPath": [_point(point) for point in BLUE_SIDE_OTHER_ROBOT_PATH],
            "reefScoringPoses": [_pose(pose) for pose in env.goal_poses],
        },
        "match": {
            "timeS": state.time_s,
            "timeRemainingS": time_remaining,
            "durationS": env.config.episode_duration_s,
            "scoredCoral": state.scored_coral,
            "maxCoral": env.config.max_coral_scored,
            "targetLevel": state.target_level,
            "targetPoints": SCORING_POINTS_TELEOP[state.target_level],
            "totalReward": state.total_reward,
            "lastReward": reward,
        },
        "robot": {
            "pose": _pose(state.pose),
            "vxMps": state.vx_mps,
            "vyMps": state.vy_mps,
            "omegaRadps": state.omega_radps,
            "hasCoral": state.has_coral,
            "isIntaking": state.is_intaking,
            "isScoring": state.is_scoring,
            "intakeProgressS": state.intake_progress_s,
            "scoreProgressS": state.score_progress_s,
            "frozenTimeS": state.frozen_time_s,
            "smoothnessReward": state.smoothness_reward,
        },
        "traffic": {
            "pose": _pose(state.other_robot_pose),
            "vxMps": state.other_robot_vx_mps,
            "vyMps": state.other_robot_vy_mps,
            "distanceM": state.other_robot_distance_m,
            "hit": state.hit_other_robot,
            "hardHit": state.hard_hit_other_robot,
            "hits": state.other_robot_hits,
            "hardHits": state.other_robot_hard_hits,
            "impactSpeedMps": state.other_robot_impact_speed_mps,
            "pathVariant": state.other_robot_path_variant,
            "speedScale": state.other_robot_speed_scale,
        },
        "objective": {
            "pose": _pose(objective),
            "goalPose": _pose(goal),
            "coralPose": _pose(coral),
            "goalIndex": state.current_goal_index,
            "sourceIndex": state.current_source_index,
            "distanceM": info["objective_distance_m"],
        },
        "ai": {
            "action": [float(value) for value in action],
            "intent": mental.intent,
            "focus": mental.focus,
            "confidence": mental.confidence,
            "risk": mental.risk,
            "actionEnergy": mental.action_energy,
            "attentionPose": _pose(mental.attention_pose),
            "thoughtTrace": mental.thought_trace,
        },
    }


def _pose(pose: Pose2d) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "heading": float(pose.heading)}


def _point(point: tuple[float, float]) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1])}

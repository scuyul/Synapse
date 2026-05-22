"""Demo-facing model telemetry for AdvantageScope."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import Pose2d


@dataclass(frozen=True, slots=True)
class MentalSnapshot:
    intent: str
    focus: str
    confidence: float
    risk: float
    action_energy: float
    objective_distance_m: float
    attention_pose: Pose2d
    thought_trace: str


class MentalVisualizerPublisher:
    """Publishes synthetic, explainable model-state topics for demos."""

    def __init__(self, nt_instance: object):
        try:
            from wpimath.geometry import Pose2d as WpiPose2d
        except ImportError as exc:
            raise ImportError(
                "Install dependencies first: python -m pip install -r .\\requirements.txt"
            ) from exc

        self.intent_pub = nt_instance.getStringTopic("/AI/Mental/Intent").publish()
        self.focus_pub = nt_instance.getStringTopic("/AI/Mental/Focus").publish()
        self.thought_trace_pub = nt_instance.getStringTopic(
            "/AI/Mental/ThoughtTrace"
        ).publish()
        self.confidence_pub = nt_instance.getDoubleTopic(
            "/AI/Mental/Confidence"
        ).publish()
        self.risk_pub = nt_instance.getDoubleTopic("/AI/Mental/Risk").publish()
        self.action_energy_pub = nt_instance.getDoubleTopic(
            "/AI/Mental/ActionEnergy"
        ).publish()
        self.objective_distance_pub = nt_instance.getDoubleTopic(
            "/AI/Mental/ObjectiveDistanceMeters"
        ).publish()
        self.attention_pose_pub = nt_instance.getStructTopic(
            "/AdvantageScope/AIAttentionPose", WpiPose2d
        ).publish()

    def publish(self, snapshot: MentalSnapshot) -> None:
        self.intent_pub.set(snapshot.intent)
        self.focus_pub.set(snapshot.focus)
        self.thought_trace_pub.set(snapshot.thought_trace)
        self.confidence_pub.set(snapshot.confidence)
        self.risk_pub.set(snapshot.risk)
        self.action_energy_pub.set(snapshot.action_energy)
        self.objective_distance_pub.set(snapshot.objective_distance_m)
        self.attention_pose_pub.set(_to_wpilib_pose(snapshot.attention_pose))


def build_mental_snapshot(
    env: ReefscapeEnv,
    action: Sequence[float],
    *,
    raw_action: Sequence[float] | None = None,
) -> MentalSnapshot:
    state = env.state
    if state is None:
        raise RuntimeError("Cannot visualize before env.reset().")

    objective = env.current_objective_pose()
    objective_distance = state.pose.distance_to(objective)
    other_distance = state.other_robot_distance_m
    action_energy = _action_energy(action)
    model_energy = _action_energy(raw_action) if raw_action is not None else action_energy
    risk = _clamp01((1.35 - other_distance) / 1.35)
    confidence = _clamp01(
        0.95
        - 0.35 * risk
        - 0.20 * _clamp01(state.frozen_time_s / 1.5)
        - 0.10 * model_energy
    )

    if state.hit_other_robot:
        intent = "recover from contact"
        focus = "traffic"
        attention_pose = state.other_robot_pose
    elif risk > 0.25:
        intent = "avoid defense"
        focus = "traffic"
        attention_pose = state.other_robot_pose
    elif state.is_intaking or (not state.has_coral and objective_distance < 0.35):
        intent = "acquire coral"
        focus = "intake"
        attention_pose = env.current_coral_pose()
    elif state.is_scoring or (state.has_coral and objective_distance < 0.45):
        intent = "score coral"
        focus = "reef"
        attention_pose = env.current_goal_pose()
    elif state.has_coral:
        intent = "route to reef"
        focus = "objective"
        attention_pose = objective
    else:
        intent = "route to coral station"
        focus = "objective"
        attention_pose = objective

    thought_trace = (
        f"{intent} | focus={focus} | conf={confidence:.2f} | "
        f"risk={risk:.2f} | dist={objective_distance:.2f}m"
    )
    return MentalSnapshot(
        intent=intent,
        focus=focus,
        confidence=confidence,
        risk=risk,
        action_energy=action_energy,
        objective_distance_m=objective_distance,
        attention_pose=attention_pose,
        thought_trace=thought_trace,
    )


def _action_energy(action: Sequence[float]) -> float:
    values = [abs(float(value)) for value in action[:5]]
    if not values:
        return 0.0
    translational = math.hypot(values[0], values[1]) if len(values) >= 2 else values[0]
    rotational = values[2] if len(values) >= 3 else 0.0
    mechanism = max(values[3:5], default=0.0)
    return _clamp01(
        0.55 * min(1.0, translational) + 0.25 * rotational + 0.20 * mechanism
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_wpilib_pose(pose: Pose2d):
    from wpimath.geometry import Pose2d as WpiPose2d, Rotation2d

    return WpiPose2d(pose.x, pose.y, Rotation2d(pose.heading))

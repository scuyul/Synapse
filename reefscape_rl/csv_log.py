"""AdvantageScope-style CSV table logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from reefscape_rl.env import ReefscapeEnv


CSV_COLUMNS = (
    "Timestamp",
    "/Sim/RobotPose/x",
    "/Sim/RobotPose/y",
    "/Sim/RobotPose/heading",
    "/Sim/RobotVelocity/vx",
    "/Sim/RobotVelocity/vy",
    "/Sim/RobotVelocity/omega",
    "/Sim/OtherRobotPose/x",
    "/Sim/OtherRobotPose/y",
    "/Sim/OtherRobotPose/heading",
    "/Sim/OtherRobotVelocity/vx",
    "/Sim/OtherRobotVelocity/vy",
    "/Sim/OtherRobotDistance",
    "/Sim/HitOtherRobot",
    "/Sim/HardHitOtherRobot",
    "/Sim/OtherRobotHits",
    "/Sim/OtherRobotHardHits",
    "/Sim/OtherRobotImpactSpeed",
    "/Sim/FrozenTime",
    "/Sim/CoralPose/x",
    "/Sim/CoralPose/y",
    "/Sim/GoalPose/x",
    "/Sim/GoalPose/y",
    "/Sim/HasCoral",
    "/Sim/ScoredCoral",
    "/Sim/TargetLevelCode",
    "/RL/Action/vx",
    "/RL/Action/vy",
    "/RL/Action/omega",
    "/RL/Action/intake",
    "/RL/Action/score",
    "/RL/Reward",
    "/RL/TotalReward",
    "/RL/SmoothnessReward",
    "/RL/EventCode",
    "/RL/ObjectiveDistance",
    "/Sim/IntakeProgress",
    "/Sim/ScoreProgress",
    "/Sim/IsIntaking",
    "/Sim/IsScoring",
    "/Tuning/IntakeDurationS",
    "/Tuning/ScoreDurationS",
)


LEVEL_CODES = {
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
}


class AdvantageScopeCsvLogger:
    """Writes CSV table logs using AdvantageScope's expected Timestamp + key columns."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

    def write_step(
        self,
        env: ReefscapeEnv,
        action: Sequence[float],
        reward: float,
        info: dict,
    ) -> None:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot log before env.reset().")
        coral_pose = env.current_coral_pose()
        goal_pose = env.current_goal_pose()
        self._writer.writerow(
            {
                "Timestamp": f"{info['time_s']:.3f}",
                "/Sim/RobotPose/x": f"{state.pose.x:.6f}",
                "/Sim/RobotPose/y": f"{state.pose.y:.6f}",
                "/Sim/RobotPose/heading": f"{state.pose.heading:.6f}",
                "/Sim/RobotVelocity/vx": f"{state.vx_mps:.6f}",
                "/Sim/RobotVelocity/vy": f"{state.vy_mps:.6f}",
                "/Sim/RobotVelocity/omega": f"{state.omega_radps:.6f}",
                "/Sim/OtherRobotPose/x": f"{state.other_robot_pose.x:.6f}",
                "/Sim/OtherRobotPose/y": f"{state.other_robot_pose.y:.6f}",
                "/Sim/OtherRobotPose/heading": f"{state.other_robot_pose.heading:.6f}",
                "/Sim/OtherRobotVelocity/vx": f"{state.other_robot_vx_mps:.6f}",
                "/Sim/OtherRobotVelocity/vy": f"{state.other_robot_vy_mps:.6f}",
                "/Sim/OtherRobotDistance": f"{info['other_robot_distance_m']:.6f}",
                "/Sim/HitOtherRobot": "true" if info["hit_other_robot"] else "false",
                "/Sim/HardHitOtherRobot": "true" if info["hard_hit_other_robot"] else "false",
                "/Sim/OtherRobotHits": info["other_robot_hits"],
                "/Sim/OtherRobotHardHits": info["other_robot_hard_hits"],
                "/Sim/OtherRobotImpactSpeed": f"{info['other_robot_impact_speed_mps']:.6f}",
                "/Sim/FrozenTime": f"{info['frozen_time_s']:.6f}",
                "/Sim/CoralPose/x": f"{coral_pose.x:.6f}",
                "/Sim/CoralPose/y": f"{coral_pose.y:.6f}",
                "/Sim/GoalPose/x": f"{goal_pose.x:.6f}",
                "/Sim/GoalPose/y": f"{goal_pose.y:.6f}",
                "/Sim/HasCoral": "true" if state.has_coral else "false",
                "/Sim/ScoredCoral": state.scored_coral,
                "/Sim/TargetLevelCode": LEVEL_CODES[env.config.target_level],
                "/RL/Action/vx": f"{float(action[0]):.6f}",
                "/RL/Action/vy": f"{float(action[1]):.6f}",
                "/RL/Action/omega": f"{float(action[2]):.6f}",
                "/RL/Action/intake": f"{float(action[3]):.6f}",
                "/RL/Action/score": f"{float(action[4]):.6f}",
                "/RL/Reward": f"{reward:.6f}",
                "/RL/TotalReward": f"{info['total_reward']:.6f}",
                "/RL/SmoothnessReward": f"{info['smoothness_reward']:.6f}",
                "/RL/EventCode": info["event_code"],
                "/RL/ObjectiveDistance": f"{info['objective_distance_m']:.6f}",
                "/Sim/IntakeProgress": f"{info['intake_progress_s']:.6f}",
                "/Sim/ScoreProgress": f"{info['score_progress_s']:.6f}",
                "/Sim/IsIntaking": "true" if info["is_intaking"] else "false",
                "/Sim/IsScoring": "true" if info["is_scoring"] else "false",
                "/Tuning/IntakeDurationS": f"{info['intake_duration_s']:.6f}",
                "/Tuning/ScoreDurationS": f"{info['score_duration_s']:.6f}",
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "AdvantageScopeCsvLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

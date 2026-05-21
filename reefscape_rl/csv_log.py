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
    "/Sim/FrozenTime",
    "/Sim/FuelPose/x",
    "/Sim/FuelPose/y",
    "/Sim/FuelPose/z",
    "/Sim/ActiveShots",
    "/Sim/GoalPose/x",
    "/Sim/GoalPose/y",
    "/Sim/HeldFuel",
    "/Sim/ScoredFuel",
    "/Sim/InactiveScoredFuel",
    "/Sim/MissedFuel",
    "/Sim/HubActive",
    "/Sim/MatchPhaseCode",
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
        fuel_pose = env.current_fuel_pose()
        fuel_pose3d = env.current_fuel_pose3d()
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
                "/Sim/FrozenTime": f"{info['frozen_time_s']:.6f}",
                "/Sim/FuelPose/x": f"{fuel_pose.x:.6f}",
                "/Sim/FuelPose/y": f"{fuel_pose.y:.6f}",
                "/Sim/FuelPose/z": f"{fuel_pose3d.z:.6f}",
                "/Sim/ActiveShots": info["active_shots"],
                "/Sim/GoalPose/x": f"{goal_pose.x:.6f}",
                "/Sim/GoalPose/y": f"{goal_pose.y:.6f}",
                "/Sim/HeldFuel": state.held_fuel,
                "/Sim/ScoredFuel": state.scored_fuel,
                "/Sim/InactiveScoredFuel": state.inactive_scored_fuel,
                "/Sim/MissedFuel": state.missed_fuel,
                "/Sim/HubActive": "true" if info["hub_active"] else "false",
                "/Sim/MatchPhaseCode": info["match_phase_code"],
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

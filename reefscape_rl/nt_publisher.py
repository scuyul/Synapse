"""NetworkTables publisher for live AdvantageScope visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import Pose2d as SimPose2d
from reefscape_rl.geometry import Pose3d as SimPose3d


@dataclass(slots=True)
class AdvantageScopeNtPublisher:
    """Publishes simulator state as WPILib structs over NT4.

    AdvantageScope can visualize these topics directly in the 2D Field tab by
    connecting to a local NetworkTables server and selecting the Pose2d topics.
    """

    inst: object
    robot_pose_pub: object
    fuel_pose_pub: object
    fuel_pose3d_pub: object
    fuel_pose3d_array_pub: object
    goal_pose_pub: object
    objective_pose_pub: object
    scoring_pose_array_pub: object
    reward_pub: object
    total_reward_pub: object
    held_fuel_pub: object
    scored_fuel_pub: object
    inactive_scored_fuel_pub: object
    missed_fuel_pub: object
    active_shots_pub: object
    hub_active_pub: object
    match_phase_pub: object
    intake_progress_pub: object
    score_progress_pub: object
    is_intaking_pub: object
    is_scoring_pub: object
    frozen_time_pub: object
    smoothness_reward_pub: object
    intake_duration_pub: object
    score_duration_pub: object
    intake_duration_sub: object
    score_duration_sub: object
    training_step_pub: object
    preview_return_pub: object
    preview_episode_pub: object

    @classmethod
    def start_server(cls, *, port: int = 5810) -> "AdvantageScopeNtPublisher":
        try:
            from ntcore import NetworkTableInstance
            from wpimath.geometry import Pose2d, Pose3d
        except ImportError as exc:
            raise ImportError("Install RobotPy first: python -m pip install robotpy") from exc

        inst = NetworkTableInstance.getDefault()
        inst.stopServer()
        inst.startServer(port4=port)

        intake_duration_topic = inst.getDoubleTopic("/Tuning/IntakeDurationS")
        score_duration_topic = inst.getDoubleTopic("/Tuning/ScoreDurationS")
        intake_duration_pub = intake_duration_topic.publish()
        score_duration_pub = score_duration_topic.publish()
        intake_duration_pub.set(0.25)
        score_duration_pub.set(0.25)

        return cls(
            inst=inst,
            robot_pose_pub=inst.getStructTopic("/AdvantageScope/RobotPose", Pose2d).publish(),
            fuel_pose_pub=inst.getStructTopic("/AdvantageScope/FuelPose", Pose2d).publish(),
            fuel_pose3d_pub=inst.getStructTopic(
                "/AdvantageScope/FuelPose3d", Pose3d
            ).publish(),
            fuel_pose3d_array_pub=inst.getStructArrayTopic(
                "/AdvantageScope/FuelPoses3d", Pose3d
            ).publish(),
            goal_pose_pub=inst.getStructTopic("/AdvantageScope/GoalPose", Pose2d).publish(),
            objective_pose_pub=inst.getStructTopic(
                "/AdvantageScope/ObjectivePose", Pose2d
            ).publish(),
            scoring_pose_array_pub=inst.getStructArrayTopic(
                "/AdvantageScope/HubScoringPoses", Pose2d
            ).publish(),
            reward_pub=inst.getDoubleTopic("/RL/Reward").publish(),
            total_reward_pub=inst.getDoubleTopic("/RL/TotalReward").publish(),
            held_fuel_pub=inst.getIntegerTopic("/Sim/HeldFuel").publish(),
            scored_fuel_pub=inst.getIntegerTopic("/Sim/ScoredFuel").publish(),
            inactive_scored_fuel_pub=inst.getIntegerTopic(
                "/Sim/InactiveScoredFuel"
            ).publish(),
            missed_fuel_pub=inst.getIntegerTopic("/Sim/MissedFuel").publish(),
            active_shots_pub=inst.getIntegerTopic("/Sim/ActiveShots").publish(),
            hub_active_pub=inst.getBooleanTopic("/Sim/HubActive").publish(),
            match_phase_pub=inst.getStringTopic("/Sim/MatchPhase").publish(),
            intake_progress_pub=inst.getDoubleTopic("/Sim/IntakeProgress").publish(),
            score_progress_pub=inst.getDoubleTopic("/Sim/ScoreProgress").publish(),
            is_intaking_pub=inst.getBooleanTopic("/Sim/IsIntaking").publish(),
            is_scoring_pub=inst.getBooleanTopic("/Sim/IsScoring").publish(),
            frozen_time_pub=inst.getDoubleTopic("/Sim/FrozenTime").publish(),
            smoothness_reward_pub=inst.getDoubleTopic("/RL/SmoothnessReward").publish(),
            intake_duration_pub=intake_duration_pub,
            score_duration_pub=score_duration_pub,
            intake_duration_sub=intake_duration_topic.subscribe(0.25),
            score_duration_sub=score_duration_topic.subscribe(0.25),
            training_step_pub=inst.getIntegerTopic("/RL/TrainingStep").publish(),
            preview_return_pub=inst.getDoubleTopic("/RL/PreviewEpisodeReturn").publish(),
            preview_episode_pub=inst.getIntegerTopic("/RL/PreviewEpisode").publish(),
        )

    def apply_tunables(self, env: ReefscapeEnv) -> None:
        env.config.intake_duration_s = _clamp_duration(self.intake_duration_sub.get())
        env.config.shot_period_s = _clamp_duration(self.score_duration_sub.get())
        self.intake_duration_pub.set(env.config.intake_duration_s)
        self.score_duration_pub.set(env.config.shot_period_s)

    def publish(self, env: ReefscapeEnv, *, reward: float = 0.0) -> None:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot publish before env.reset().")

        self.robot_pose_pub.set(_to_wpilib_pose(state.pose))
        self.fuel_pose_pub.set(_to_wpilib_pose(env.current_fuel_pose()))
        self.fuel_pose3d_pub.set(_to_wpilib_pose3d(env.current_fuel_pose3d()))
        self.fuel_pose3d_array_pub.set(_to_wpilib_poses3d(env.fuel_poses3d()))
        self.goal_pose_pub.set(_to_wpilib_pose(env.current_goal_pose()))
        self.objective_pose_pub.set(_to_wpilib_pose(env.current_objective_pose()))
        self.scoring_pose_array_pub.set(_to_wpilib_poses(env.goal_poses))
        self.reward_pub.set(float(reward))
        self.total_reward_pub.set(float(state.total_reward))
        self.held_fuel_pub.set(int(state.held_fuel))
        self.scored_fuel_pub.set(int(state.scored_fuel))
        self.inactive_scored_fuel_pub.set(int(state.inactive_scored_fuel))
        self.missed_fuel_pub.set(int(state.missed_fuel))
        self.active_shots_pub.set(int(len(state.fuel_shots)))
        self.hub_active_pub.set(bool(env.is_blue_hub_active()))
        self.match_phase_pub.set(str(env._match_phase_name(env._match_phase_code(state.time_s))))
        self.intake_progress_pub.set(float(state.intake_progress_s))
        self.score_progress_pub.set(float(state.score_progress_s))
        self.is_intaking_pub.set(bool(state.is_intaking))
        self.is_scoring_pub.set(bool(state.is_scoring))
        self.frozen_time_pub.set(float(state.frozen_time_s))
        self.smoothness_reward_pub.set(float(state.smoothness_reward))
        self.inst.flush()

    def publish_training(
        self,
        *,
        training_step: int,
        preview_episode_return: float,
        preview_episode: int,
    ) -> None:
        self.training_step_pub.set(int(training_step))
        self.preview_return_pub.set(float(preview_episode_return))
        self.preview_episode_pub.set(int(preview_episode))
        self.inst.flush()


def _to_wpilib_pose(pose: SimPose2d):
    from wpimath.geometry import Pose2d, Rotation2d

    return Pose2d(pose.x, pose.y, Rotation2d(pose.heading))


def _to_wpilib_pose3d(pose: SimPose3d):
    from wpimath.geometry import Pose3d, Rotation3d

    return Pose3d(pose.x, pose.y, pose.z, Rotation3d(pose.roll, pose.pitch, pose.yaw))


def _to_wpilib_poses3d(poses: Sequence[SimPose3d]):
    return [_to_wpilib_pose3d(pose) for pose in poses]


def _to_wpilib_poses(poses: Sequence[SimPose2d]):
    return [_to_wpilib_pose(pose) for pose in poses]


def _clamp_duration(value: float) -> float:
    return max(0.05, min(5.0, float(value)))

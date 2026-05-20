"""NetworkTables publisher for live AdvantageScope visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reefscape_rl.env import ReefscapeEnv
from reefscape_rl.geometry import Pose2d as SimPose2d


@dataclass(slots=True)
class AdvantageScopeNtPublisher:
    """Publishes simulator state as WPILib structs over NT4.

    AdvantageScope can visualize these topics directly in the 2D Field tab by
    connecting to a local NetworkTables server and selecting the Pose2d topics.
    """

    inst: object
    robot_pose_pub: object
    other_robot_pose_pub: object
    coral_pose_pub: object
    goal_pose_pub: object
    objective_pose_pub: object
    reef_pose_array_pub: object
    reward_pub: object
    total_reward_pub: object
    has_coral_pub: object
    scored_coral_pub: object
    intake_progress_pub: object
    score_progress_pub: object
    is_intaking_pub: object
    is_scoring_pub: object
    other_robot_distance_pub: object
    hit_other_robot_pub: object
    hard_hit_other_robot_pub: object
    other_robot_hits_pub: object
    other_robot_hard_hits_pub: object
    other_robot_impact_speed_pub: object
    frozen_time_pub: object
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
            from wpimath.geometry import Pose2d
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
            other_robot_pose_pub=inst.getStructTopic(
                "/AdvantageScope/OtherRobotPose", Pose2d
            ).publish(),
            coral_pose_pub=inst.getStructTopic("/AdvantageScope/CoralPose", Pose2d).publish(),
            goal_pose_pub=inst.getStructTopic("/AdvantageScope/GoalPose", Pose2d).publish(),
            objective_pose_pub=inst.getStructTopic(
                "/AdvantageScope/ObjectivePose", Pose2d
            ).publish(),
            reef_pose_array_pub=inst.getStructArrayTopic(
                "/AdvantageScope/ReefScoringPoses", Pose2d
            ).publish(),
            reward_pub=inst.getDoubleTopic("/RL/Reward").publish(),
            total_reward_pub=inst.getDoubleTopic("/RL/TotalReward").publish(),
            has_coral_pub=inst.getBooleanTopic("/Sim/HasCoral").publish(),
            scored_coral_pub=inst.getIntegerTopic("/Sim/ScoredCoral").publish(),
            intake_progress_pub=inst.getDoubleTopic("/Sim/IntakeProgress").publish(),
            score_progress_pub=inst.getDoubleTopic("/Sim/ScoreProgress").publish(),
            is_intaking_pub=inst.getBooleanTopic("/Sim/IsIntaking").publish(),
            is_scoring_pub=inst.getBooleanTopic("/Sim/IsScoring").publish(),
            other_robot_distance_pub=inst.getDoubleTopic(
                "/Sim/OtherRobotDistance"
            ).publish(),
            hit_other_robot_pub=inst.getBooleanTopic("/Sim/HitOtherRobot").publish(),
            hard_hit_other_robot_pub=inst.getBooleanTopic(
                "/Sim/HardHitOtherRobot"
            ).publish(),
            other_robot_hits_pub=inst.getIntegerTopic("/Sim/OtherRobotHits").publish(),
            other_robot_hard_hits_pub=inst.getIntegerTopic(
                "/Sim/OtherRobotHardHits"
            ).publish(),
            other_robot_impact_speed_pub=inst.getDoubleTopic(
                "/Sim/OtherRobotImpactSpeed"
            ).publish(),
            frozen_time_pub=inst.getDoubleTopic("/Sim/FrozenTime").publish(),
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
        env.config.score_duration_s = _clamp_duration(self.score_duration_sub.get())
        self.intake_duration_pub.set(env.config.intake_duration_s)
        self.score_duration_pub.set(env.config.score_duration_s)

    def publish(self, env: ReefscapeEnv, *, reward: float = 0.0) -> None:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot publish before env.reset().")

        self.robot_pose_pub.set(_to_wpilib_pose(state.pose))
        self.other_robot_pose_pub.set(_to_wpilib_pose(state.other_robot_pose))
        self.coral_pose_pub.set(_to_wpilib_pose(env.current_coral_pose()))
        self.goal_pose_pub.set(_to_wpilib_pose(env.current_goal_pose()))
        self.objective_pose_pub.set(_to_wpilib_pose(env.current_objective_pose()))
        self.reef_pose_array_pub.set(_to_wpilib_poses(env.goal_poses))
        self.reward_pub.set(float(reward))
        self.total_reward_pub.set(float(state.total_reward))
        self.has_coral_pub.set(bool(state.has_coral))
        self.scored_coral_pub.set(int(state.scored_coral))
        self.intake_progress_pub.set(float(state.intake_progress_s))
        self.score_progress_pub.set(float(state.score_progress_s))
        self.is_intaking_pub.set(bool(state.is_intaking))
        self.is_scoring_pub.set(bool(state.is_scoring))
        self.other_robot_distance_pub.set(float(state.other_robot_distance_m))
        self.hit_other_robot_pub.set(bool(state.hit_other_robot))
        self.hard_hit_other_robot_pub.set(bool(state.hard_hit_other_robot))
        self.other_robot_hits_pub.set(int(state.other_robot_hits))
        self.other_robot_hard_hits_pub.set(int(state.other_robot_hard_hits))
        self.other_robot_impact_speed_pub.set(float(state.other_robot_impact_speed_mps))
        self.frozen_time_pub.set(float(state.frozen_time_s))
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


def _to_wpilib_poses(poses: Sequence[SimPose2d]):
    return [_to_wpilib_pose(pose) for pose in poses]


def _clamp_duration(value: float) -> float:
    return max(0.05, min(5.0, float(value)))

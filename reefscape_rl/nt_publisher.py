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
    coral_pose_pub: object
    goal_pose_pub: object
    objective_pose_pub: object
    reef_pose_array_pub: object
    reward_pub: object
    total_reward_pub: object
    has_coral_pub: object
    scored_coral_pub: object

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

        return cls(
            inst=inst,
            robot_pose_pub=inst.getStructTopic("/AdvantageScope/RobotPose", Pose2d).publish(),
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
        )

    def publish(self, env: ReefscapeEnv, *, reward: float = 0.0) -> None:
        state = env.state
        if state is None:
            raise RuntimeError("Cannot publish before env.reset().")

        self.robot_pose_pub.set(_to_wpilib_pose(state.pose))
        self.coral_pose_pub.set(_to_wpilib_pose(env.current_coral_pose()))
        self.goal_pose_pub.set(_to_wpilib_pose(env.current_goal_pose()))
        self.objective_pose_pub.set(_to_wpilib_pose(env.current_objective_pose()))
        self.reef_pose_array_pub.set(_to_wpilib_poses(env.goal_poses))
        self.reward_pub.set(float(reward))
        self.total_reward_pub.set(float(state.total_reward))
        self.has_coral_pub.set(bool(state.has_coral))
        self.scored_coral_pub.set(int(state.scored_coral))
        self.inst.flush()


def _to_wpilib_pose(pose: SimPose2d):
    from wpimath.geometry import Pose2d, Rotation2d

    return Pose2d(pose.x, pose.y, Rotation2d(pose.heading))


def _to_wpilib_poses(poses: Sequence[SimPose2d]):
    return [_to_wpilib_pose(pose) for pose in poses]

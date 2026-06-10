// Copyright 2021-2024 FRC 6328
// http://github.com/Mechanical-Advantage
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// version 3 as published by the Free Software Foundation or
// available in the root directory of this project.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

package frc.robot.Robot24.subsystems.drive;

import static edu.wpi.first.units.Units.*;

import com.pathplanner.lib.auto.AutoBuilder;
import com.pathplanner.lib.config.ModuleConfig;
import com.pathplanner.lib.config.PIDConstants;
import com.pathplanner.lib.config.RobotConfig;
import com.pathplanner.lib.controllers.PPHolonomicDriveController;
import com.pathplanner.lib.pathfinding.Pathfinding;
import com.pathplanner.lib.util.PathPlannerLogging;
import edu.wpi.first.hal.FRCNetComm.tInstances;
import edu.wpi.first.hal.FRCNetComm.tResourceType;
import edu.wpi.first.hal.HAL;
import edu.wpi.first.math.estimator.SwerveDrivePoseEstimator;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.math.kinematics.SwerveDriveKinematics;
import edu.wpi.first.math.kinematics.SwerveModulePosition;
import edu.wpi.first.math.kinematics.SwerveModuleState;
import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj.DriverStation;
import edu.wpi.first.wpilibj.DriverStation.Alliance;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import edu.wpi.first.wpilibj2.command.sysid.SysIdRoutine;
import frc.robot.Robot24.util.LocalADStarAK;
import frc.robot.SimConstants;
import frc.robot.SimConstants.Mode;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.ironmaple.simulation.drivesims.COTS;
import org.ironmaple.simulation.drivesims.configs.DriveTrainSimulationConfig;
import org.ironmaple.simulation.drivesims.configs.SwerveModuleSimulationConfig;
import org.littletonrobotics.junction.AutoLogOutput;
import org.littletonrobotics.junction.Logger;

public class Drive extends SubsystemBase /* implements Vision.VisionConsumer */ {

  // Configure path planner
  private static final RobotConfig PP_CONFIG =
      new RobotConfig(DriveConstants.ROBOT_MASS_KG, DriveConstants.ROBOT_MOI,
          new ModuleConfig(DriveConstants.FrontLeft.WheelRadius,
              DriveConstants.kSpeedAt12Volts.in(MetersPerSecond), DriveConstants.WHEEL_COF,
              DCMotor.getKrakenX60(1).withReduction(DriveConstants.FrontLeft.DriveMotorGearRatio),
              DriveConstants.FrontLeft.SlipCurrent, 1),
          getModuleTranslations());

  // Maple Sim config constants
  public static final DriveTrainSimulationConfig MAPLE_SIM_CONFIG =
      DriveTrainSimulationConfig.Default().withRobotMass(Kilograms.of(DriveConstants.ROBOT_MASS_KG))
          .withCustomModuleTranslations(getModuleTranslations()).withGyro(COTS.ofPigeon2())
          .withSwerveModule(new SwerveModuleSimulationConfig(DCMotor.getKrakenX60(1),
              DCMotor.getKrakenX60(1), DriveConstants.FrontLeft.DriveMotorGearRatio,
              DriveConstants.FrontLeft.SteerMotorGearRatio,
              Volts.of(DriveConstants.FrontLeft.DriveFrictionVoltage),
              Volts.of(DriveConstants.FrontLeft.SteerFrictionVoltage),
              Meters.of(DriveConstants.FrontLeft.WheelRadius),
              KilogramSquareMeters.of(DriveConstants.FrontLeft.SteerInertia),
              DriveConstants.WHEEL_COF));

  static final Lock odometryLock = new ReentrantLock();
  private final GyroIO gyroIO;
  private final GyroIOInputsAutoLogged gyroInputs = new GyroIOInputsAutoLogged();
  private final Module[] modules = new Module[4]; // FL, FR, BL, BR
  private final SysIdRoutine sysId;
  private final Alert gyroDisconnectedAlert =
      new Alert("Disconnected gyro, using kinematics as fallback.", AlertType.kError);

  private SwerveDriveKinematics kinematics = new SwerveDriveKinematics(getModuleTranslations());
  private Rotation2d rawGyroRotation = new Rotation2d();
  private SwerveModulePosition[] lastModulePositions = // For delta tracking
      new SwerveModulePosition[] {new SwerveModulePosition(), new SwerveModulePosition(),
          new SwerveModulePosition(), new SwerveModulePosition()};
  private SwerveDrivePoseEstimator poseEstimator =
      new SwerveDrivePoseEstimator(kinematics, rawGyroRotation, lastModulePositions, Pose2d.kZero);

  private boolean snapToRotationEnabled = false;
  private Rotation2d desiredRotation = new Rotation2d();

  public Drive(GyroIO gyroIO, ModuleIO flModuleIO, ModuleIO frModuleIO, ModuleIO blModuleIO,
      ModuleIO brModuleIO) {
    this.gyroIO = gyroIO;
    modules[0] = new Module(flModuleIO, 0, DriveConstants.FrontLeft);
    modules[1] = new Module(frModuleIO, 1, DriveConstants.FrontRight);
    modules[2] = new Module(blModuleIO, 2, DriveConstants.BackLeft);
    modules[3] = new Module(brModuleIO, 3, DriveConstants.BackRight);

    // Usage reporting for swerve template
    HAL.report(tResourceType.kResourceType_RobotDrive, tInstances.kRobotDriveSwerve_AdvantageKit);

    // Start odometry thread
    PhoenixOdometryThread.getInstance().start();

    // Configure AutoBuilder for PathPlanner
    AutoBuilder.configure(this::getPose, this::setPose, this::getChassisSpeeds, this::runVelocity,
        new PPHolonomicDriveController(new PIDConstants(0.001, 0.0, 0.0),
            new PIDConstants(2.0, 0.0, 0.3)),
        PP_CONFIG, () -> DriverStation.getAlliance().orElse(Alliance.Blue) == Alliance.Red, this);
    Pathfinding.setPathfinder(new LocalADStarAK());
    PathPlannerLogging.setLogActivePathCallback((activePath) -> {
      Logger.recordOutput("Odometry/Trajectory", activePath.toArray(new Pose2d[activePath.size()]));
    });
    PathPlannerLogging.setLogTargetPoseCallback((targetPose) -> {
      Logger.recordOutput("Odometry/TrajectorySetpoint", targetPose);
    });

    // Configure SysId
    sysId = new SysIdRoutine(
        new SysIdRoutine.Config(null, null, null,
            (state) -> Logger.recordOutput("Drive/SysIdState", state.toString())),
        new SysIdRoutine.Mechanism((voltage) -> runCharacterization(voltage.in(Volts)), null,
            this));
  }

  @Override
  public void periodic() {
    odometryLock.lock(); // Prevents odometry updates while reading data
    gyroIO.updateInputs(gyroInputs);
    Logger.processInputs("Drive/Gyro", gyroInputs);
    for (var module : modules) {
      module.periodic();
    }
    odometryLock.unlock();

    // Stop moving when disabled
    if (DriverStation.isDisabled()) {
      for (var module : modules) {
        module.stop();
      }
    }

    // Log empty setpoint states when disabled
    if (DriverStation.isDisabled()) {
      Logger.recordOutput("SwerveStates/Setpoints", new SwerveModuleState[] {});
      Logger.recordOutput("SwerveStates/SetpointsOptimized", new SwerveModuleState[] {});
    }

    // Update odometry
    double[] sampleTimestamps = modules[0].getOdometryTimestamps(); // All signals are sampled
                                                                    // together
    int sampleCount = sampleTimestamps.length;
    for (int i = 0; i < sampleCount; i++) {
      // Read wheel positions and deltas from each module
      SwerveModulePosition[] modulePositions = new SwerveModulePosition[4];
      SwerveModulePosition[] moduleDeltas = new SwerveModulePosition[4];
      for (int moduleIndex = 0; moduleIndex < 4; moduleIndex++) {
        modulePositions[moduleIndex] = modules[moduleIndex].getOdometryPositions()[i];
        moduleDeltas[moduleIndex] = new SwerveModulePosition(
            modulePositions[moduleIndex].distanceMeters
                - lastModulePositions[moduleIndex].distanceMeters,
            modulePositions[moduleIndex].angle);
        lastModulePositions[moduleIndex] = modulePositions[moduleIndex];
      }

      rawGyroRotation = gyroInputs.odometryYawPositions[i];

      // Apply update
      poseEstimator.updateWithTime(sampleTimestamps[i], rawGyroRotation, modulePositions);
    }

    // Update gyro alert
    gyroDisconnectedAlert.set(!gyroInputs.connected && SimConstants.CURRENT_MODE != Mode.SIM);
  }

  /**
   * Runs the drive at the desired velocity.
   *
   * @param speeds Speeds in meters/sec
   */
  public void runVelocity(ChassisSpeeds speeds) {
    // Calculate module setpoints
    speeds = ChassisSpeeds.discretize(speeds, TimedRobot.kDefaultPeriod);
    SwerveModuleState[] setpointStates = kinematics.toSwerveModuleStates(speeds);
    SwerveDriveKinematics.desaturateWheelSpeeds(setpointStates, DriveConstants.kSpeedAt12Volts);

    // Log unoptimized setpoints and setpoint speeds
    Logger.recordOutput("SwerveStates/Setpoints", setpointStates);
    Logger.recordOutput("SwerveChassisSpeeds/Setpoints", speeds);

    // Send setpoints to modules
    for (int i = 0; i < 4; i++) {
      modules[i].runSetpoint(setpointStates[i]);
    }

    // Log optimized setpoints (runSetpoint mutates each state)
    Logger.recordOutput("SwerveStates/SetpointsOptimized", setpointStates);
  }

  /** Runs the drive in a straight line with the specified drive output. */
  public void runCharacterization(double output) {
    for (int i = 0; i < 4; i++) {
      modules[i].runCharacterization(output);
    }
  }

  /** Stops the drive. */
  public void stop() {
    runVelocity(new ChassisSpeeds());
  }

  /**
   * Stops the drive and turns the modules to an X arrangement to resist movement. The modules will
   * return to their normal orientations the next time a nonzero velocity is requested.
   */
  public void stopWithX() {
    Rotation2d[] headings = new Rotation2d[4];
    for (int i = 0; i < 4; i++) {
      headings[i] = getModuleTranslations()[i].getAngle();
    }
    kinematics.resetHeadings(headings);
    stop();
  }

  public void setSnapToRotation(boolean enabled) {
    snapToRotationEnabled = enabled;
  }

  public boolean getSnapToRotation() {
    return snapToRotationEnabled;
  }

  public void setDesiredRotation(Rotation2d rotation) {
    desiredRotation = rotation;
    setSnapToRotation(true);
  }

  public Rotation2d getDesiredRotation() {
    return desiredRotation;
  }

  /** Returns a command to run a quasistatic test in the specified direction. */
  public Command sysIdQuasistatic(SysIdRoutine.Direction direction) {
    return run(() -> runCharacterization(0.0)).withTimeout(1.0)
        .andThen(sysId.quasistatic(direction));
  }

  /** Returns a command to run a dynamic test in the specified direction. */
  public Command sysIdDynamic(SysIdRoutine.Direction direction) {
    return run(() -> runCharacterization(0.0)).withTimeout(1.0).andThen(sysId.dynamic(direction));
  }

  /**
   * Returns the module states (turn angles and drive velocities) for all of the modules.
   */
  @AutoLogOutput(key = "SwerveStates/Measured")
  private SwerveModuleState[] getModuleStates() {
    SwerveModuleState[] states = new SwerveModuleState[4];
    for (int i = 0; i < 4; i++) {
      states[i] = modules[i].getState();
    }
    return states;
  }

  /**
   * Returns the module positions (turn angles and drive positions) for all of the modules.
   */
  private SwerveModulePosition[] getModulePositions() {
    SwerveModulePosition[] states = new SwerveModulePosition[4];
    for (int i = 0; i < 4; i++) {
      states[i] = modules[i].getPosition();
    }
    return states;
  }

  /** Returns the measured chassis speeds of the robot. */
  @AutoLogOutput(key = "SwerveChassisSpeeds/Measured")
  private ChassisSpeeds getChassisSpeeds() {
    return kinematics.toChassisSpeeds(getModuleStates());
  }

  /** Returns the position of each module in radians. */
  public double[] getWheelRadiusCharacterizationPositions() {
    double[] values = new double[4];
    for (int i = 0; i < 4; i++) {
      values[i] = modules[i].getWheelRadiusCharacterizationPosition();
    }
    return values;
  }

  /**
   * Returns the average velocity of the modules in rotations/sec (Phoenix native units).
   */
  public double getFFCharacterizationVelocity() {
    double output = 0.0;
    for (int i = 0; i < 4; i++) {
      output += modules[i].getFFCharacterizationVelocity() / 4.0;
    }
    return output;
  }

  /** Returns the current odometry pose. */
  @AutoLogOutput(key = "Odometry/Robot")
  public Pose2d getPose() {
    return poseEstimator.getEstimatedPosition();
  }

  /** Returns the current odometry rotation. */
  public Rotation2d getRotation() {
    return getPose().getRotation();
  }

  /** Resets the current odometry pose. */
  public void setPose(Pose2d pose) {
    poseEstimator.resetPosition(rawGyroRotation, getModulePositions(), pose);
  }

  // /** Adds a new timestamped vision measurement. */
  // @Override
  // public void accept(
  // Pose2d visionRobotPoseMeters,
  // double timestampSeconds,
  // Matrix<N3, N1> visionMeasurementStdDevs) {
  // poseEstimator.addVisionMeasurement(
  // visionRobotPoseMeters, timestampSeconds, visionMeasurementStdDevs);
  // }

  /** Returns the maximum linear speed in meters per sec. */
  public double getMaxLinearSpeedMetersPerSec() {
    return DriveConstants.kSpeedAt12Volts.in(MetersPerSecond);
  }

  /** Returns the maximum angular speed in radians per sec. */
  public double getMaxAngularSpeedRadPerSec() {
    return getMaxLinearSpeedMetersPerSec() / DriveConstants.DRIVE_BASE_RADIUS;
  }

  /** Returns an array of module translations. */
  public static Translation2d[] getModuleTranslations() {
    return new Translation2d[] {
        new Translation2d(DriveConstants.FrontLeft.LocationX, DriveConstants.FrontLeft.LocationY),
        new Translation2d(DriveConstants.FrontRight.LocationX, DriveConstants.FrontRight.LocationY),
        new Translation2d(DriveConstants.BackLeft.LocationX, DriveConstants.BackLeft.LocationY),
        new Translation2d(DriveConstants.BackRight.LocationX, DriveConstants.BackRight.LocationY)};
  }
}

// Copyright Copyright Copyright Copyright

package frc.robot.Robot25;

import static edu.wpi.first.units.Units.*;
import com.ctre.phoenix6.swerve.SwerveModuleConstants;
import com.ctre.phoenix6.swerve.SwerveModuleConstants.DriveMotorArrangement;
import com.ctre.phoenix6.swerve.SwerveModuleConstants.SteerMotorArrangement;
import com.pathplanner.lib.auto.AutoBuilder;
import com.pathplanner.lib.auto.NamedCommands;
import com.pathplanner.lib.commands.PathPlannerAuto;

import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Transform2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.wpilibj.DriverStation;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.CommandScheduler;
import edu.wpi.first.wpilibj2.command.Commands;
import edu.wpi.first.wpilibj2.command.button.CommandXboxController;
import edu.wpi.first.wpilibj2.command.sysid.SysIdRoutine;
import frc.robot.Robot25.commands.DriveCharacterization;
import frc.robot.Robot25.commands.DriveCommands;
import frc.robot.Robot25.commands.DriveCommands.ReefPositions;
import frc.robot.Robot25.subsystems.AlgaeEater.Algae;
import frc.robot.Robot25.subsystems.AlgaeEater.AlgaeIO;
import frc.robot.Robot25.subsystems.AlgaeEater.AlgaeIOSim;
import frc.robot.Robot25.subsystems.AlgaeEater.AlgaeIOTalonFX;
import frc.robot.Robot25.subsystems.drive.Drive;
import frc.robot.Robot25.subsystems.drive.DriveConstants;
import frc.robot.Robot25.subsystems.drive.ModuleIO;
import frc.robot.Robot25.subsystems.drive.ModuleIOSim;
import frc.robot.Robot25.subsystems.drive.ModuleIOTalonFX;
import frc.robot.Robot25.subsystems.elevator.Elevator;
import frc.robot.Robot25.subsystems.elevator.Elevator.Level;
import frc.robot.Robot25.subsystems.elevator.ElevatorIO;
import frc.robot.Robot25.subsystems.elevator.ElevatorIOSim;
import frc.robot.Robot25.subsystems.elevator.ElevatorIOTalonFXNew;
import frc.robot.Robot25.subsystems.gyro.GyroIO;
import frc.robot.Robot25.subsystems.gyro.GyroIOPigeon2;
import frc.robot.Robot25.subsystems.gyro.GyroIOSim;
import frc.robot.Robot25.subsystems.outtake.Outtake;
import frc.robot.Robot25.subsystems.outtake.OuttakeIO;
import frc.robot.Robot25.subsystems.outtake.OuttakeIOSim;
import frc.robot.Robot25.subsystems.outtake.OuttakeIOTalonFX;
import frc.robot.Robot25.subsystems.vision.Vision;
import frc.robot.Robot25.subsystems.vision.VisionConstants;
import frc.robot.Robot25.subsystems.vision.VisionIO;
import frc.robot.Robot25.subsystems.vision.VisionIOLimelight;
import frc.robot.Robot25.subsystems.vision.VisionIOPhotonVisionSim;
import java.util.Set;
import frc.robot.SimConstants;
import org.ironmaple.simulation.drivesims.SwerveDriveSimulation;
import org.littletonrobotics.junction.AutoLogOutput;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.networktables.LoggedDashboardChooser;

public class RobotContainer extends frc.lib.RobotContainer {

  // Subsystems
  private final Drive drive;
  private final Elevator elevator;
  private final Outtake outtake;
  private final Vision vision;
  private final Algae algae;

  // Drive simulation
  private static final SwerveDriveSimulation driveSimulation =
      new SwerveDriveSimulation(Drive.MAPLE_SIM_CONFIG, SimConstants.SIM_INITIAL_FIELD_POSE);
  private final CommandXboxController driverController = new CommandXboxController(0);
  private final CommandXboxController operatorController = new CommandXboxController(1);

  private final LoggedDashboardChooser<Command> autoChooser;

  @AutoLogOutput
  public final Pose3d[] mechanismPoses = new Pose3d[] {Pose3d.kZero, Pose3d.kZero, Pose3d.kZero,};

  public RobotContainer() {
    super(driveSimulation);
    // Check for valid swerve config
    var modules = new SwerveModuleConstants[] {DriveConstants.FrontLeft, DriveConstants.FrontRight,
        DriveConstants.BackLeft, DriveConstants.BackRight,};
    for (var constants : modules) {
      if (constants.DriveMotorType != DriveMotorArrangement.TalonFX_Integrated
          || constants.SteerMotorType != SteerMotorArrangement.TalonFX_Integrated) {
        throw new RuntimeException(
            "You are using an unsupported swerve configuration, which this template does not support without manual customization. The 2025 release of Phoenix supports some swerve configurations which were not available during 2025 beta testing, preventing any development and support from the AdvantageKit developers.");
      }
    }

    switch (SimConstants.CURRENT_MODE) {
      case REAL:
        // Real robot, instantiate hardware IO implementations
        drive = new Drive(new GyroIOPigeon2(), new ModuleIOTalonFX(DriveConstants.FrontLeft),
            new ModuleIOTalonFX(DriveConstants.FrontRight),
            new ModuleIOTalonFX(DriveConstants.BackLeft),
            new ModuleIOTalonFX(DriveConstants.BackRight), driveSimulation::setSimulationWorldPose);

        elevator = new Elevator(new ElevatorIOTalonFXNew());
        outtake = new Outtake(new OuttakeIOTalonFX());
        vision = new Vision(drive,
            new VisionIOLimelight("limelight-front", () -> drive.getPose().getRotation())
        // new VisionIOLimelight("limelight-back", () -> drive.getPose().getRotation())
        );
        algae = new Algae(new AlgaeIOTalonFX());
        break;
      case SIM:
        drive = new Drive(new GyroIOSim(driveSimulation.getGyroSimulation()),
            new ModuleIOSim(driveSimulation.getModules()[0]),
            new ModuleIOSim(driveSimulation.getModules()[1]),
            new ModuleIOSim(driveSimulation.getModules()[2]),
            new ModuleIOSim(driveSimulation.getModules()[3]),
            driveSimulation::setSimulationWorldPose);
        drive.setPose(SimConstants.SIM_INITIAL_FIELD_POSE);

        elevator = new Elevator(new ElevatorIOSim());
        outtake = new Outtake(new OuttakeIOSim());
        vision = new Vision(drive, new VisionIOPhotonVisionSim(VisionConstants.limelightBackName,
            VisionConstants.limelightBackTransform, driveSimulation::getSimulatedDriveTrainPose),
            new VisionIOPhotonVisionSim(VisionConstants.limelightFrontName,
                VisionConstants.limelightFrontTransform,
                driveSimulation::getSimulatedDriveTrainPose));
        algae = new Algae(new AlgaeIOSim());
        break;
      default:
        // Replayed robot, disable IO implementations
        drive = new Drive(new GyroIO() {}, new ModuleIO() {}, new ModuleIO() {}, new ModuleIO() {},
            new ModuleIO() {}, driveSimulation::setSimulationWorldPose);

        elevator = new Elevator(new ElevatorIO() {});

        outtake = new Outtake(new OuttakeIO() {});

        vision = new Vision(drive, new VisionIO() {});
        algae = new Algae(new AlgaeIO() {});
        break;
    }

    // Values are tuned to speed but may be changed
    NamedCommands.registerCommand("Exhaust", outtake.depositCoral());
    NamedCommands.registerCommand("Auto1", DriveCommands.FirstSnapper(drive).withTimeout(2));
    NamedCommands.registerCommand("Align1", DriveCommands.AutoSnapper(drive).withTimeout(2));
    NamedCommands.registerCommand("Align2", DriveCommands.AutoSnapper(drive).withTimeout(2.5));
    NamedCommands.registerCommand("RightSource", DriveCommands.SourceSnapper(drive).withTimeout(2));
    NamedCommands.registerCommand("LeftSource", DriveCommands.SourceSnapper(drive).withTimeout(2));
    NamedCommands.registerCommand("AlgaeSnapper", DriveCommands.AlgaeSnapper(drive).withTimeout(2));
    // Probably dont change or use
    NamedCommands.registerCommand("Align3", DriveCommands.FirstSnapper(drive).withTimeout(1));
    NamedCommands.registerCommand("Align4", DriveCommands.AutoSnapper(drive).withTimeout(1.5));
    NamedCommands.registerCommand("RightSource2",
        DriveCommands.AutoSourceRight(drive).withTimeout(1));
    NamedCommands.registerCommand("LeftSource2",
        DriveCommands.AutoSourceLeft(drive).withTimeout(1));

    NamedCommands.registerCommand("L0", elevator.L0()
        .andThen(outtake.autoQueueCoral2().until(outtake.seesAtOutputTrigger.debounce(0.1))));
    NamedCommands.registerCommand("L1", elevator.L1());
    NamedCommands.registerCommand("L2", elevator.L2());
    NamedCommands.registerCommand("L3", elevator.L3());
    NamedCommands.registerCommand("L4", elevator.L4());
    NamedCommands.registerCommand("Algae", elevator.Algae());
    NamedCommands.registerCommand("Ground", elevator.intakeHeight());
    NamedCommands.registerCommand("Intake",
        outtake.autoQueueCoral3().until(outtake.seesAtOutputTrigger.debounce(0.1)));

    NamedCommands.registerCommand("CO.ScoreFirstL4",
        DriveCommands.FullSnapperOuterAuto(drive).alongWith(elevator.L0())
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral())
            .andThen(DriveCommands.FullSnapperOuter(drive).alongWith(elevator.L0())));
    NamedCommands.registerCommand("CO.ScoreL4",
        DriveCommands.FullSnapperOuterAuto(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral())
            .andThen(DriveCommands.FullSnapperOuter(drive).alongWith(elevator.L0())));
    NamedCommands.registerCommand("CO.LoadCoral", DriveCommands.SourceSnapper(drive).withTimeout(2)
        .andThen(outtake.autoQueueCoral(false).until(outtake.seesAtOutputTrigger.debounce(0.1))));

    NamedCommands.registerCommand("Maybe1",
        DriveCommands.FullSnapperOuter(drive).alongWith(elevator.L0())
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral())
            .andThen(DriveCommands.FullSnapperOuter(drive).alongWith(elevator.L0())));
    NamedCommands.registerCommand("Maybe2", (DriveCommands.SourceSnapper(drive).withTimeout(3))
        .andThen(outtake.autoQueueCoral3().until(outtake.seesAtOutputTrigger.debounce(0.1))));
    NamedCommands.registerCommand("Maybe3",
        DriveCommands.FullSnapperOuterAuto(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral()));
    NamedCommands.registerCommand("Maybe4",
        (elevator.L0().alongWith(DriveCommands.SourceSnapper(drive).withTimeout(2)))
            .andThen(outtake.autoQueueCoral3().until(outtake.seesAtOutputTrigger.debounce(0.1))));
    NamedCommands.registerCommand("Maybe5",
        DriveCommands.FullSnapperOuter(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral()).andThen(elevator.L0()));

    NamedCommands.registerCommand("MN.3C.R1", DriveCommands.FlySnappyV2(drive));
    NamedCommands.registerCommand("MN.3C.R2", DriveCommands.SourceSnapper(drive));

    NamedCommands.registerCommand("MN.1C.M1", DriveCommands.BargeSnapper(drive));

    NamedCommands.registerCommand("AlgaeIntake", algae.setOpenLoop(Volts.of(-6)).withTimeout(3));
    NamedCommands.registerCommand("AlgaeOuttake", algae.setOpenLoop(Volts.of(10)).withTimeout(1));
    NamedCommands.registerCommand("AlgaeHeight", elevator.openLoop(() -> .5).withTimeout(3));
    NamedCommands.registerCommand("AlgaeSnap", DriveCommands.AlgaeSnapper(drive).withTimeout(3));
    NamedCommands.registerCommand("BargeHeight", elevator.Algae());

    NamedCommands.registerCommand("SnapReef.A",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.A));
    NamedCommands.registerCommand("SnapReef.B",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.B));
    NamedCommands.registerCommand("SnapReef.C",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.C));
    NamedCommands.registerCommand("SnapReef.D",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.D));
    NamedCommands.registerCommand("SnapReef.E",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.E));
    NamedCommands.registerCommand("SnapReef.F",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.F));
    NamedCommands.registerCommand("SnapReef.G",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.G));
    NamedCommands.registerCommand("SnapReef.H",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.H));
    NamedCommands.registerCommand("SnapReef.I",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.I));
    NamedCommands.registerCommand("SnapReef.J",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.J));
    NamedCommands.registerCommand("SnapReef.K",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.K));
    NamedCommands.registerCommand("SnapReef.L",
        DriveCommands.FlySnappyV2Named(drive, ReefPositions.L));
    // Set up auto routines
    autoChooser =
        new LoggedDashboardChooser<>("Auto Choices", AutoBuilder.buildAutoChooser("AL.0C.1M"));

    autoChooser.addOption("Static Drive Voltage", Commands.run(() -> drive.driveOpenLoop(10)));
    autoChooser.addOption("Static Turn Voltage", Commands.run(() -> drive.TurnOpenLoop(10)));

    // Set up SysId routines
    autoChooser.addOption("Drive Wheel Radius Characterization",
        DriveCharacterization.wheelRadiusCharacterization(drive));
    autoChooser.addOption("Drive Simple FF Characterization",
        DriveCharacterization.feedforwardCharacterization(drive));
    autoChooser.addOption("Drive SysId (Quasistatic Forward)",
        drive.sysIdQuasistatic(SysIdRoutine.Direction.kForward));
    autoChooser.addOption("Drive SysId (Quasistatic Reverse)",
        drive.sysIdQuasistatic(SysIdRoutine.Direction.kReverse));
    autoChooser.addOption("Drive SysId (Dynamic Forward)",
        drive.sysIdDynamic(SysIdRoutine.Direction.kForward));
    autoChooser.addOption("Drive SysId (Dynamic Reverse)",
        drive.sysIdDynamic(SysIdRoutine.Direction.kReverse));

    configureButtonBindings();
  }

  private void configureButtonBindings() {

    // Due to field orientation, joystick Y (forward) controls X direction and vice
    // versa
    drive.setDefaultCommand(DriveCommands.joystickDriveAssist(drive,
        () -> driverController.getLeftY(), () -> driverController.getLeftX(),
        () -> -driverController.getRightX() * .85, driverController.leftTrigger(),
        driverController.rightTrigger(0.15).or(elevator.isAtHeight(Level.L4))));
    outtake.setDefaultCommand(
        outtake.autoQueueCoral(false).onlyWhile(elevator.isAtHeight(Level.Intake))
            .withName("RobotContainer.outtakeDefaultCommand"));
    driverController.start().onTrue(Commands.runOnce(() -> {
      drive.setPose(new Pose2d(drive.getPose().getTranslation(), Rotation2d.kZero));
      CommandScheduler.getInstance().cancelAll(); // clear out any commands that might be stuck
      if (DriverStation.isEnabled()) {
        drive.startIgnoringVision();
      }


    }, drive).ignoringDisable(true).withName("RobotContainer.driverZeroCommand"));
    driverController.rightBumper().whileTrue(DriveCommands.FlySnappyV2(drive));
    driverController.leftBumper().whileTrue(DriveCommands.FLYSnappySource(drive));

    driverController.x().whileTrue(DriveCommands.AlgaeSnapper(drive));
    driverController.y()
        .whileTrue(DriveCommands.FullSnapperOuter(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L4()))
            .andThen(outtake.depositCoral()).andThen(elevator.L0()));
    driverController.b()
        .whileTrue(DriveCommands.FullSnapperOuter(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L3()))
            .andThen(outtake.depositCoral()).andThen(elevator.L0()));
    driverController.a()
        .whileTrue(DriveCommands.FullSnapperOuter(drive)
            .andThen(DriveCommands.FullSnapperInner(drive).alongWith(elevator.L2()))
            .andThen(outtake.depositCoral()).andThen(elevator.L0()));

    driverController.povLeft().onTrue(DriveCommands.FlySnappyV2Left(drive));
    driverController.povUp().whileTrue(DriveCommands.BargeSnapper(drive));
    driverController.povRight().onTrue(DriveCommands.FlySnappyV2Right(drive));
    driverController.povDown().onTrue(DriveCommands.snapToRotation(drive));

    // OPERATOR CONTROLs
    operatorController.leftTrigger().whileTrue(outtake.autoQueueCoralOveride());
    operatorController.rightTrigger().whileTrue(outtake.reverseCoral());
    operatorController.start().onTrue(elevator.zeroElevator());
    operatorController.back().onTrue(elevator.zeroElevator());
    operatorController.leftBumper().onTrue(outtake.depositCoral().andThen(elevator.intakeHeight())
        .withName("RobotContainer.intakeHeight"));
    operatorController.rightBumper().onTrue(elevator.intakeHeight());
    operatorController.a().onTrue(elevator.L2());
    operatorController.x().onTrue(elevator.Algae());
    operatorController.b().onTrue(elevator.L3());
    operatorController.y().onTrue(elevator.L4());
    operatorController.povRight().whileTrue(algae.setOpenLoop(Volts.of(10)));
    operatorController.povLeft().whileTrue(algae.setOpenLoop(Volts.of(-6)));
    operatorController.povDown().onTrue(algae.setOpenLoop(Volts.of(10)));
    operatorController.povUp().onTrue(algae.setOpenLoop(Volts.of(-6)));

    operatorController.axisMagnitudeGreaterThan(1, 0.1)
        .whileTrue(outtake.openLoop(operatorController::getLeftY));

    operatorController.axisMagnitudeGreaterThan(5, 0.1)
        .whileTrue(elevator.openLoop(operatorController::getRightY));
    // ##########################################################################################################
  }

  @Override
  public Command getAutonomousCommand() {
    return autoChooser.get();

  }

  public Command getTestCommand() {
    return DriveCommands.testCommand(drive);

  }

  @Override
  public void disabledInit() {}

  @Override
  public void disabledPeriodic() {
    var autoCommand = autoChooser.get();

    if (autoCommand instanceof PathPlannerAuto) {
      var auto = (PathPlannerAuto) autoCommand;
      Pose2d startPose = auto.getStartingPose();
      Logger.recordOutput("AutoStartPose", startPose);
    }

  }

  @Override
  public void robotPeriodic() {
    var elevatorPoses = elevator.getElevatorPoses();
    mechanismPoses[0] = elevatorPoses[0];
    mechanismPoses[1] = elevatorPoses[1];
    mechanismPoses[2] = elevatorPoses[2];
  }

  @Override
  public void simulationPeriodic() {}
}

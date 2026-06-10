package frc.robot.Robot25.commands;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj2.command.Command;
import frc.robot.Robot25.subsystems.drive.Drive;
import frc.robot.Robot25.subsystems.elevator.Elevator;
import frc.robot.Robot25.subsystems.outtake.Outtake;
import org.littletonrobotics.junction.Logger;

public class AiModeCommand extends Command {
  private final Drive drive;
  private final Elevator elevator;
  private final Outtake outtake;
  private final NetworkTableEntry enabledEntry;
  private final NetworkTableEntry vxEntry;
  private final NetworkTableEntry vyEntry;
  private final NetworkTableEntry omegaEntry;
  private final NetworkTableEntry intakeEntry;
  private final NetworkTableEntry scoreEntry;
  private final NetworkTableEntry scoreLevelEntry;
  private boolean previousIntakeRequest = false;
  private boolean previousScoreRequest = false;

  public AiModeCommand(Drive drive, Elevator elevator, Outtake outtake) {
    this.drive = drive;
    this.elevator = elevator;
    this.outtake = outtake;
    NetworkTable table = NetworkTableInstance.getDefault().getTable("AI").getSubTable("Command");
    enabledEntry = table.getEntry("enabled");
    vxEntry = table.getEntry("vx");
    vyEntry = table.getEntry("vy");
    omegaEntry = table.getEntry("omega");
    intakeEntry = table.getEntry("intake");
    scoreEntry = table.getEntry("score");
    scoreLevelEntry = table.getEntry("scoreLevel");
    addRequirements(drive);
    setName("AiModeCommand");
  }

  @Override
  public void initialize() {
    previousIntakeRequest = false;
    previousScoreRequest = false;
    enabledEntry.setBoolean(true);
    drive.setSnapToRotation(false);
    Logger.recordOutput("AI/ModeActive", true);
  }

  @Override
  public void execute() {
    double vxNorm = clampCommand(vxEntry.getDouble(0.0));
    double vyNorm = clampCommand(vyEntry.getDouble(0.0));
    double omegaNorm = clampCommand(omegaEntry.getDouble(0.0));

    ChassisSpeeds speeds = new ChassisSpeeds(vxNorm * drive.getMaxLinearSpeedMetersPerSec(),
        vyNorm * drive.getMaxLinearSpeedMetersPerSec(),
        omegaNorm * drive.getMaxAngularSpeedRadPerSec());
    drive.runVelocity(ChassisSpeeds.fromFieldRelativeSpeeds(speeds, drive.getRotation()));

    boolean intakeRequest = intakeEntry.getDouble(0.0) > 0.5;
    boolean scoreRequest = scoreEntry.getDouble(0.0) > 0.5;
    if (intakeRequest && !previousIntakeRequest) {
      elevator.intakeHeight().andThen(outtake.autoQueueCoral3()).schedule();
    }
    if (scoreRequest && !previousScoreRequest) {
      scoreCommandForRequestedLevel().andThen(outtake.depositCoral()).andThen(elevator.intakeHeight())
          .schedule();
    }
    previousIntakeRequest = intakeRequest;
    previousScoreRequest = scoreRequest;

    Logger.recordOutput("AI/Command/vx", vxNorm);
    Logger.recordOutput("AI/Command/vy", vyNorm);
    Logger.recordOutput("AI/Command/omega", omegaNorm);
    Logger.recordOutput("AI/Command/intake", intakeRequest);
    Logger.recordOutput("AI/Command/score", scoreRequest);
    Logger.recordOutput("AI/Command/scoreLevel", requestedScoreLevel());
  }

  @Override
  public void end(boolean interrupted) {
    enabledEntry.setBoolean(false);
    drive.stop();
    Logger.recordOutput("AI/ModeActive", false);
  }

  private double clampCommand(double value) {
    return MathUtil.clamp(value, -1.0, 1.0);
  }

  private Command scoreCommandForRequestedLevel() {
    switch (requestedScoreLevel()) {
      case 1:
        return elevator.L1();
      case 2:
        return elevator.L2();
      case 3:
        return elevator.L3();
      default:
        return elevator.L4();
    }
  }

  private int requestedScoreLevel() {
    return (int) Math.round(MathUtil.clamp(scoreLevelEntry.getDouble(4.0), 1.0, 4.0));
  }
}

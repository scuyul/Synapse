package frc.robot.Robot25.subsystems.elevator;

import static edu.wpi.first.units.Units.Inches;
import static edu.wpi.first.units.Units.Meters;
import static edu.wpi.first.units.Units.Radians;
import static edu.wpi.first.units.Units.Volts;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.*;

import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.units.measure.Angle;
import edu.wpi.first.units.measure.Distance;
import edu.wpi.first.wpilibj.util.Color;
import edu.wpi.first.wpilibj.util.Color8Bit;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import edu.wpi.first.wpilibj2.command.button.Trigger;
import java.util.function.DoubleSupplier;
import org.littletonrobotics.junction.AutoLogOutput;
import org.littletonrobotics.junction.Logger;
import org.littletonrobotics.junction.mechanism.LoggedMechanism2d;
import org.littletonrobotics.junction.mechanism.LoggedMechanismLigament2d;
import org.littletonrobotics.junction.mechanism.LoggedMechanismRoot2d;

public class Elevator extends SubsystemBase {

  private final ElevatorIO io;

  private final ElevatorIOInputsAutoLogged inputs = new ElevatorIOInputsAutoLogged();

  @AutoLogOutput
  public final LoggedMechanism2d mechanism2d =
      new LoggedMechanism2d(3, 3, new Color8Bit(Color.kBlack));

  private final LoggedMechanismRoot2d mechRoot2d = mechanism2d.getRoot("Elevator Root", 1.5, 0);
  private final LoggedMechanismLigament2d elevatorMech2d =
      mechRoot2d.append(new LoggedMechanismLigament2d("Elevator", INITIAL_HEIGHT.in(Meters), 90.0,
          50, new Color8Bit(Color.kBlue)));
  // private WantedState wantedState = WantedState.minHeight;
  // private SystemState systemState = SystemState.zeroingMinhight;

  // private enum WantedState {
  // minHeight, L1, l2, l3, l4, maxHeight
  // }

  // private enum SystemState {
  // zeroingMinhight, minHeight, L1, l2, l3, l4, maxHeight
  // }

  public enum Level {

    // original value during W0 = L4 = 72 + 6

    minHeight(MIN_HEIGHT), Intake(Inches.of(16.4 + 3.5)), L1(Inches.of(18 + 14)), L2(
        Inches.of(38.66)), L3(Inches.of(54.5)), L4(Inches.of(78.25)), Barge(Inches.of(82));

    private final Distance height;

    Level(Distance height) {
      this.height = height;
    }

    public Distance getHeight() {
      return height;
    }

    public Level up() {
      switch (this) {
        case minHeight:
          return L1;
        case L1:
          return L2;
        case L2:
          return L3;
        case L3:
          return L4;
        default:
          return L4;
      }
    }

    public Level down() {
      switch (this) {
        case L4:
          return L3;
        case L3:
          return L2;
        case L2:
          return L1;
        case L1:
          return minHeight;
        default:
          return minHeight;
      }
    }

  };

  private Level desiredLevel = Level.minHeight;

  public Elevator(ElevatorIO io) {
    this.io = io;

    /*
     * When the lower limit is hit, set the Elevator's state (where we believe we're at) to
     * minHeight and set the winch to 0 volts
     */
    lowerLimitHit.onTrue(Commands.runOnce(() -> {
      System.out.println(
          "[Elevator] Lower limit hit, setting state to minHeight and setting motor volts to 0");
      desiredLevel = Level.minHeight;
      io.setWinchOpenLoop(Volts.of(0));
      io.zeroEncoder();

    }).ignoringDisable(true).withName("Elevator.lowerLimitHit"));

    // currentLevel = Level.minHeight;
    // Distance height = currentLevel.getHeight();
    // Angle r = inchesToRadians(height);
    // io.setWinchPosition(r);
    goToLevel(Level.minHeight);
  }

  @Override
  public void periodic() {
    io.updateInputs(inputs);
    Logger.processInputs("Elevator", inputs);
    elevatorMech2d.setLength(radiansToInches(inputs.winchPosition).in(Meters));

    Logger.recordOutput("Elevator/EstimatedHeight",
        radiansToInches(inputs.winchPosition).in(Meters));

    Logger.recordOutput("Elevator/CurrentLevel", desiredLevel);
    Logger.recordOutput("Elevator/CurrentLevelHeight", desiredLevel.getHeight());
    Logger.recordOutput("Elevator/isAtGoal", isAtGoal());
    Logger.recordOutput("Elevator/lowerLimitHIt", lowerLimitHit);
  }

  private Angle inchesToRadians(Distance d) {
    // d.minus(MIN_HEIGHT); // does nothing
    return Radians.of(d.minus(MIN_HEIGHT).in(Meters) / DRUM_RADIUS.in(Meters));
  }

  private Distance radiansToInches(Angle a) {
    double d = a.in(Radians) * DRUM_RADIUS.in(Meters);
    return Meters.of(d).plus(MIN_HEIGHT);
  }

  public Command goToLevel(Level level) {
    return this.runOnce(() -> {
      desiredLevel = level;
      Distance height = level.getHeight();
      Angle r = inchesToRadians(height);
      io.setWinchPosition(r);
    }).andThen(Commands.waitUntil(isAtGoal()).withName("Elevator.goToLevel_" + level.name()));
  }

  public Command minHeight() {
    return this.runOnce(() -> {
      desiredLevel = Level.minHeight;
      Distance height = Level.minHeight.getHeight();
      Angle r = inchesToRadians(height);
      io.setWinchPosition(r);
    })
        // .andThen(Commands.runOnce(() -> io.setWinchOpenLoop(Volts.of(-9))))
        .andThen(Commands.waitUntil(lowerLimitHit)).andThen(Commands.runOnce(() -> {
          // io.zeroEncoder();
          io.setWinchOpenLoop(Volts.of(0));
        })).withName("Elevator.minHeight");
  }

  public Command miniHeight() {
    return this.runOnce(() -> {
      desiredLevel = Level.minHeight;
      Distance height = Level.minHeight.getHeight();
      Angle r = inchesToRadians(height);
      io.setWinchPosition(r);

    }).andThen(Commands.waitUntil(isAtGoal()))
        // .andThen(Commands.runOnce(() -> io.setWinchOpenLoop(Volts.of(-9))))
        .andThen(Commands.waitUntil(lowerLimitHit)).andThen(Commands.runOnce(() -> {
          // io.zeroEncoder();
          io.setWinchOpenLoop(Volts.of(0));
        })).withName("Elevator.miniHeight");
  }

  // public Command miniHeight() {
  // return goToLevel(Level.minHeight)
  // .andThen(Commands.runOnce(() -> io.setWinchOpenLoop(Volts.of(-10))));

  // }
  public Command zeroElevator() {
    return Commands.runOnce(() -> io.zeroEncoder()).withName("Elevator.zeroElevator");
  }

  // src/main/java/frc/robot/Robot25/subsystems/elevator/Elevator.java

  public Command L0() {
    return goToLevel(Level.Intake);
  }

  public Command intakeHeight() {
    return goToLevel(Level.Intake);
  }

  public Command L1() {
    return goToLevel(Level.L1);
  }

  public Command L2() {
    return goToLevel(Level.L2);
  }

  public Command L3() {
    return goToLevel(Level.L3);
  }

  public Command L4() {
    return goToLevel(Level.L4);
  }

  public Command maxHeight() {
    return goToLevel(Level.L4);
  }

  public Command Algae() {
    return goToLevel(Level.Barge);
  }

  public Command upLevel() {
    return this.runOnce(() -> {
      desiredLevel = desiredLevel.up();
      Distance height = desiredLevel.getHeight();
      Angle r = inchesToRadians(height);
      io.setWinchPosition(r);
    }).withName("Elevator.upLevel");
  }

  public Command downLevel() {
    return this.runOnce(() -> {
      desiredLevel = desiredLevel.down();
      Distance height = desiredLevel.getHeight();
      Angle r = inchesToRadians(height);
      io.setWinchPosition(r);
    }).withName("Elevator.downLevel");
  }

  public Command openLoop(DoubleSupplier speed) {
    return this.runEnd(() -> {
      io.setWinchOpenLoop(Volts.of(speed.getAsDouble() * -4));
    }, () -> {
      io.setWinchPosition(inputs.winchPosition);
    }).withName("Elevator.openLoop");
  }

  // public Command upLevel() {
  // return goToLevel(currentLevel.up());
  // }

  // public Command downLevel() {
  // return goToLevel(currentLevel.down());
  // }

  /**
   * Computes the estimated position of each elevator stage (stage 1, stage 2, carriage + loader)
   * given the current winch position. Useful for rendering the robot model in simulation.
   *
   * A continuous elevator lifts stages in order of least weight. As of 2/1, for us that would be
   * stage 2, stage 1, then carriage. In order, each stage will contribute as much height as it can
   * to the lift before the next stage engages.
   *
   * @returns Pose array of each elevator stage in the order: stage 1, stage 2, carriage
   */
  public Pose3d[] getElevatorPoses() {
    var height = radiansToInches(inputs.winchPosition).minus(MIN_HEIGHT);
    var heightInches = height.in(Inches);
    // contributes no height until above 26.25 inches and contributes up to 25.25
    // inches
    var stage1Contribution = heightInches < 26.25 ? 0
        : heightInches < 51.5 ? height.minus(Inches.of(26.25)).in(Meters)
            : Inches.of(25.25).in(Meters);
    // contributes up to 26.25 inches starting at height 0
    var stage2Contribution = heightInches < 26.25 ? height.in(Meters) : Inches.of(26.25).in(Meters);
    // contributes no height until above 51.5 inches and contributes up to 25.25
    // inches
    var stage3Contribution = heightInches < 51.5 ? 0 : height.minus(Inches.of(51.5)).in(Meters);

    return new Pose3d[] {new Pose3d(0, 0, stage1Contribution, Rotation3d.kZero),
        new Pose3d(0, 0, stage1Contribution + stage2Contribution, Rotation3d.kZero), new Pose3d(0,
            0, stage1Contribution + stage2Contribution + stage3Contribution, Rotation3d.kZero),};
  }

  public final Trigger lowerLimitHit = new Trigger(() -> inputs.lowerLimit);

  public Trigger isAtHeight(Level desiredLevel) {

    return isAtGoal().and(() -> desiredLevel == this.desiredLevel);

  }

  // todo CHECK VALID TOLERANCE
  public Trigger isAtGoal() {
    return new Trigger(() -> {
      return Math.abs((inputs.winchPosition.in(Radians)
          - inchesToRadians(desiredLevel.getHeight()).in(Radians))) < 1;

    });
  }

}

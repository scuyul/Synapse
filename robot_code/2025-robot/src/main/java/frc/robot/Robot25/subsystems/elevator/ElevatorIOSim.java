package frc.robot.Robot25.subsystems.elevator;

import static edu.wpi.first.units.Units.Amps;
import static edu.wpi.first.units.Units.Kilograms;
import static edu.wpi.first.units.Units.Meters;
import static edu.wpi.first.units.Units.MetersPerSecond;
import static edu.wpi.first.units.Units.Radians;
import static edu.wpi.first.units.Units.RadiansPerSecond;
import static edu.wpi.first.units.Units.RadiansPerSecondPerSecond;
import static edu.wpi.first.units.Units.Seconds;
import static edu.wpi.first.units.Units.Volts;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.*;

import org.ironmaple.simulation.motorsims.MapleMotorSim;
import org.ironmaple.simulation.motorsims.SimMotorConfigs;
import org.ironmaple.simulation.motorsims.SimulatedMotorController;
import org.littletonrobotics.junction.Logger;
import edu.wpi.first.math.controller.ElevatorFeedforward;
import edu.wpi.first.math.controller.ProfiledPIDController;
import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.math.trajectory.TrapezoidProfile;
import edu.wpi.first.units.measure.Angle;
import edu.wpi.first.units.measure.Voltage;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj.simulation.ElevatorSim;
import frc.robot.Robot25.subsystems.elevator.ElevatorConstants.Sim;

public class ElevatorIOSim implements ElevatorIO {

  private final SimulatedMotorController.GenericMotorController winchMotorController;
  private final MapleMotorSim elevatorMotor;

  private static final DCMotor elevatorGearbox = DCMotor.getKrakenX60(2);

  private boolean winchClosedLoop = false;
  private Voltage winchAppliedVoltage = Volts.of(0);
  // private Voltage winchFFVoltage = Volts.of(0);

  private final ProfiledPIDController pidController =
      new ProfiledPIDController(Sim.kP, Sim.kI, Sim.kD,
          new TrapezoidProfile.Constraints(
              MAX_VELOCITY.in(MetersPerSecond) / DRUM_RADIUS.in(Meters),
              MAX_ACCELERATION.in(RadiansPerSecondPerSecond)));

  private final ElevatorFeedforward feedForwardController =
      new ElevatorFeedforward(Sim.kS, Sim.kG, Sim.kV, Sim.kA);

  private final ElevatorSim elevatorSim =
      new ElevatorSim(elevatorGearbox, GEARING, CARRIAGE_MASS.in(Kilograms), DRUM_RADIUS.in(Meters),
          0, MAX_EXTENSION.in(Meters), true, 0, 0.000015, 0);

  public ElevatorIOSim() {
    elevatorMotor = new MapleMotorSim(
        new SimMotorConfigs(elevatorGearbox, GEARING, Sim.MOTOR_LOAD_MOI, Sim.FRICTION_VOLTAGE));
    winchMotorController =
        elevatorMotor.useSimpleDCMotorController().withCurrentLimit(CURRENT_LIMIT);
  }

  @Override
  public void updateInputs(ElevatorIOInputs inputs) {
    var winchPositionRads = elevatorSim.getPositionMeters() / DRUM_RADIUS.in(Meters);

    if (winchClosedLoop) {
      var pidVolts = pidController.calculate(winchPositionRads);
      var pidSetpoint = pidController.getSetpoint();
      var feedForwardVolts = feedForwardController.calculate(pidSetpoint.velocity);
      winchAppliedVoltage = Volts.of(feedForwardVolts + pidVolts);
      Logger.recordOutput("Elevator/Tuning/DesiredVelocity", pidSetpoint.velocity);
    } else {
      pidController.reset(winchPositionRads);
    }

    winchMotorController.requestVoltage(winchAppliedVoltage);
    elevatorSim.setInputVoltage(elevatorMotor.getAppliedVoltage().in(Volts));

    elevatorMotor.update(Seconds.of(TimedRobot.kDefaultPeriod));
    elevatorSim.update(TimedRobot.kDefaultPeriod);

    // Update motor inputs
    inputs.winchConnected = true;
    inputs.winchPosition = Radians.of(winchPositionRads);
    inputs.winchVelocity =
        RadiansPerSecond.of(elevatorSim.getVelocityMetersPerSecond() / DRUM_RADIUS.in(Meters));
    inputs.winchAppliedVolts = winchAppliedVoltage;
    inputs.winchCurrent = Amps.of(elevatorSim.getCurrentDrawAmps());
  }

  @Override
  public void setWinchOpenLoop(Voltage output) {
    winchAppliedVoltage = output;
    winchClosedLoop = false;
  }

  @Override
  public void setWinchPosition(Angle angle) {
    pidController.setGoal(angle.in(Radians));
    winchClosedLoop = true;
  }

  @Override
  public void zeroEncoder() {

    // TODO
  }
}

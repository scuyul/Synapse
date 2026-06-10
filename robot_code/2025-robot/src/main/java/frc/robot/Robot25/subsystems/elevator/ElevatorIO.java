package frc.robot.Robot25.subsystems.elevator;

import static edu.wpi.first.units.Units.Amps;
import static edu.wpi.first.units.Units.Radians;
import static edu.wpi.first.units.Units.RadiansPerSecond;
import static edu.wpi.first.units.Units.Volts;

import org.littletonrobotics.junction.AutoLog;

import edu.wpi.first.units.measure.Angle;
import edu.wpi.first.units.measure.AngularVelocity;
import edu.wpi.first.units.measure.Current;
import edu.wpi.first.units.measure.Voltage;

public interface ElevatorIO {
  @AutoLog
  public static class ElevatorIOInputs {
    public boolean lowerLimit = false;
    public boolean winchConnected = false;
    public Angle winchPosition = Radians.of(0.0);
    public AngularVelocity winchVelocity = RadiansPerSecond.of(0.0);
    public Voltage winchAppliedVolts = Volts.of(0.0);
    public Current winchCurrent = Amps.of(0.0);
  }

  /** Updates the set of loggable inputs. */
  public default void updateInputs(ElevatorIOInputs inputs) {}

  /** Run the drive side at the specified open loop value. */
  public default void setWinchOpenLoop(Voltage output) {}

  /** Run the drive side at the specified velocity. */
  public default void setWinchPosition(Angle angle) {}

  public default void zeroEncoder() {

  }

}

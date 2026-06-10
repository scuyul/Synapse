package frc.robot.Robot25.subsystems.outtake;

import static edu.wpi.first.units.Units.Amps;
import static edu.wpi.first.units.Units.RadiansPerSecond;
import static edu.wpi.first.units.Units.Volts;

import edu.wpi.first.units.measure.AngularVelocity;
import edu.wpi.first.units.measure.Current;
import edu.wpi.first.units.measure.Voltage;
import org.littletonrobotics.junction.AutoLog;

public interface OuttakeIO {

  @AutoLog
  public static class OuttakeIOInputs {
    public boolean outtakeConnected = false;
    public AngularVelocity outtakeVelocity = RadiansPerSecond.of(0.0);
    public Voltage outtakeAppliedVolts = Volts.of(0.0);
    public Current outtakeCurrent = Amps.of(0.0);

    public boolean seesCoralAtInput = false;
    public boolean seesCoralAtOutput = false;
  }

  /** Updates the set of loggable inputs. */
  public default void updateInputs(OuttakeIOInputs inputs) {}

  /** Run the drive side at the specified open loop value. */
  public default void setOpenLoop(Voltage output) {}

  public default void setRollerOpenLoop(Voltage OuttakeOutput) {}

}

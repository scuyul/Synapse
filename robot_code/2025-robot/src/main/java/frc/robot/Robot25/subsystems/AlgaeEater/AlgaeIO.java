package frc.robot.Robot25.subsystems.AlgaeEater;

import static edu.wpi.first.units.Units.*;
import edu.wpi.first.units.measure.AngularVelocity;
import edu.wpi.first.units.measure.Current;
import edu.wpi.first.units.measure.Voltage;
import org.littletonrobotics.junction.AutoLog;

public interface AlgaeIO {



  @AutoLog
  public static class AlgaeIOInputs {
    public boolean algaeConnected = false;
    public AngularVelocity algaeVelocity = RadiansPerSecond.of(0.0);
    public Voltage algaeAppliedVolts = Volts.of(0.0);
    public Current algaeCurrent = Amps.of(0.0);

  }

  /** Updates the set of loggable inputs. */
  public default void updateInputs(AlgaeIOInputs inputs) {}

  /** Run the drive side at the specified open loop value. */
  public default void setOpenLoop(Voltage output) {}

}

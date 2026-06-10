package frc.robot.Robot25.subsystems.AlgaeEater;

import static edu.wpi.first.units.Units.*;
import com.ctre.phoenix6.signals.NeutralModeValue;
import edu.wpi.first.units.measure.Voltage;
import frc.lib.devices.TalonFXWrapper;

public class AlgaeIOTalonFX implements AlgaeIO {
  private final TalonFXWrapper algaeTalonFX;
  private final int motorID = 25;

  public AlgaeIOTalonFX() {
    algaeTalonFX = new TalonFXWrapper(motorID, "Algae", true, NeutralModeValue.Brake);
  }

  @Override
  public void setOpenLoop(Voltage output) {
    algaeTalonFX.setVoltageOut(output);
  }

  @Override
  public void updateInputs(AlgaeIOInputs inputs) {
    inputs.algaeVelocity = algaeTalonFX.getVelocity();
    inputs.algaeCurrent = algaeTalonFX.getTorqueCurrent();
    inputs.algaeAppliedVolts = Volts.of(0);
  }
}

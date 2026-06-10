package frc.lib.selfCheck;

import edu.wpi.first.units.measure.Angle;
import frc.lib.devices.TalonFXWrapper;

public class PositionTalonCheck extends CheckCommand {
  TalonFXWrapper talon;
  Angle position;
  Angle tolerance;

  public PositionTalonCheck(TalonFXWrapper talon, Angle position, Angle tolerance) {
    this.talon = talon;
    this.position = position;
    this.tolerance = tolerance;
  }

  @Override
  public void execute() {
    talon.setMotionMagicVoltage(position);
  }

  @Override
  public double getTimeoutSeconds() {
    return 5;
  }

  @Override
  public String getDescription() {
    return "set Position Of " + talon.getName() + " to " + position.toLongString();
  }

  @Override
  public boolean isFinished() {
    return talon.isAtPositionReference(position, tolerance);
  }
}

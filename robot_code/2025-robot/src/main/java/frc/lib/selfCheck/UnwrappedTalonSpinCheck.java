package frc.lib.selfCheck;

import com.ctre.phoenix6.controls.DutyCycleOut;
import com.ctre.phoenix6.hardware.TalonFX;

public class UnwrappedTalonSpinCheck extends CheckCommand {
  TalonFX talon;
  double position;
  boolean isForward;
  String name;

  public UnwrappedTalonSpinCheck(String name, TalonFX talon, boolean isForward) {
    this.talon = talon;
    this.isForward = isForward;
    this.name = name;
  }

  @Override
  public void initialize() {
    position = talon.getPosition().getValueAsDouble();
  }

  @Override
  public boolean isFinished() {
    return Math.abs(position - talon.getPosition().getValueAsDouble()) > 10;
  }

  @Override
  public double getTimeoutSeconds() {
    return 3;
  }

  @Override
  public void end(boolean interrupted) {
    talon.setControl(new DutyCycleOut(0));
  }

  @Override
  public void execute() {
    talon.setControl(new DutyCycleOut(isForward ? 0.25 : -0.25));
  }

  @Override
  public String getDescription() {
    return "spin " + name + (isForward ? "forward" : "backward");
  }
}

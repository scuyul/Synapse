package frc.robot.Robot25.subsystems.elevator;

import static edu.wpi.first.units.Units.Amps;
import static edu.wpi.first.units.Units.Meters;
import static edu.wpi.first.units.Units.MetersPerSecond;
import static edu.wpi.first.units.Units.RadiansPerSecond;
import static edu.wpi.first.units.Units.RotationsPerSecond;
import static edu.wpi.first.units.Units.RotationsPerSecondPerSecond;
import static edu.wpi.first.units.Units.Volts;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.CURRENT_LIMIT;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.DRUM_RADIUS;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.GEARING;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.MAX_ACCELERATION;
import static frc.robot.Robot25.subsystems.elevator.ElevatorConstants.MAX_VELOCITY;
import com.ctre.phoenix6.BaseStatusSignal;
import com.ctre.phoenix6.StatusSignal;
import com.ctre.phoenix6.configs.Slot0Configs;
import com.ctre.phoenix6.configs.TalonFXConfiguration;
import com.ctre.phoenix6.controls.Follower;
import com.ctre.phoenix6.controls.MotionMagicVoltage;
import com.ctre.phoenix6.hardware.ParentDevice;
import com.ctre.phoenix6.hardware.TalonFX;
import com.ctre.phoenix6.signals.FeedbackSensorSourceValue;
import com.ctre.phoenix6.signals.InvertedValue;
import com.ctre.phoenix6.signals.NeutralModeValue;
import edu.wpi.first.math.filter.Debouncer;
import edu.wpi.first.units.measure.Angle;
import edu.wpi.first.units.measure.AngularVelocity;
import edu.wpi.first.units.measure.Current;
import edu.wpi.first.units.measure.Voltage;
import frc.lib.devices.DigitalInputWrapper;
import frc.robot.Robot25.subsystems.elevator.ElevatorConstants.Real;
import org.littletonrobotics.junction.Logger;

public class ElevatorIOTalonFXNew implements ElevatorIO {

  private final DigitalInputWrapper lowerLimitSwitch =
      new DigitalInputWrapper(3, "lowerLimit", true);
  private final TalonFX lead, follower;
  private final StatusSignal<Angle> leadPosition;
  private final StatusSignal<AngularVelocity> leadVelocity;
  private final StatusSignal<Voltage> leadVoltage;
  private final StatusSignal<Current> leadCurrent;
  private final int rightMotorID = 20;
  private final int leftMotorID = 21;
  private final Debouncer leadConnectedDebouncer = new Debouncer(.5);

  // Current PIDS
  private Slot0Configs currentPids = new Slot0Configs();

  public ElevatorIOTalonFXNew() {

    lead = new TalonFX(rightMotorID);
    follower = new TalonFX(leftMotorID);

    var leadConfig = new TalonFXConfiguration();

    leadVelocity = lead.getVelocity();
    leadPosition = lead.getPosition();
    leadVoltage = lead.getMotorVoltage();
    leadCurrent = lead.getStatorCurrent();

    leadConfig.MotorOutput.Inverted = InvertedValue.CounterClockwise_Positive;
    leadConfig.Feedback.FeedbackSensorSource = FeedbackSensorSourceValue.RotorSensor;

    leadConfig.CurrentLimits.SupplyCurrentLimit = CURRENT_LIMIT.in(Amps);
    leadConfig.CurrentLimits.SupplyCurrentLimitEnable = true;
    leadConfig.Feedback.SensorToMechanismRatio = GEARING;

    leadConfig.Voltage.PeakForwardVoltage = 10;
    leadConfig.Voltage.PeakReverseVoltage = -10;

    leadConfig.MotionMagic.MotionMagicAcceleration =
        MAX_ACCELERATION.in(RotationsPerSecondPerSecond);
    leadConfig.MotionMagic.MotionMagicCruiseVelocity = RadiansPerSecond
        .of(MAX_VELOCITY.in(MetersPerSecond) / DRUM_RADIUS.in(Meters)).in(RotationsPerSecond);

    leadConfig.MotorOutput.NeutralMode = NeutralModeValue.Brake;

    leadConfig.Slot0 = currentPids;
    Real.kP.addListener(kP -> {
      currentPids.kP = kP;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kI.addListener(kI -> {
      currentPids.kI = kI;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kD.addListener(kD -> {
      currentPids.kD = kD;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kS.addListener(kS -> {
      currentPids.kS = kS;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kG.addListener(kG -> {
      currentPids.kG = kG;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kV.addListener(kV -> {
      currentPids.kV = kV;
      lead.getConfigurator().apply(currentPids);
    });
    Real.kA.addListener(kA -> {
      currentPids.kA = kA;
      lead.getConfigurator().apply(currentPids);
    });

    lead.getConfigurator().apply(leadConfig, 0.25);
    lead.setPosition(0);
    follower.setControl(new Follower(rightMotorID, true));

    BaseStatusSignal.setUpdateFrequencyForAll(50, leadCurrent, leadVoltage, leadPosition,
        leadVelocity);
    ParentDevice.optimizeBusUtilizationForAll(lead, follower);

  }

  @Override
  public void setWinchOpenLoop(Voltage output) {
    lead.setVoltage(output.in(Volts));
  }

  @Override
  public void setWinchPosition(Angle angle) {
    Logger.recordOutput("elevetorAngle", angle);
    lead.setControl(new MotionMagicVoltage(angle));
  }

  @Override
  public void zeroEncoder() {
    lead.setPosition(0);
  }

  @Override
  public void updateInputs(ElevatorIOInputs inputs) {
    var connectedStatus =
        BaseStatusSignal.refreshAll(leadCurrent, leadVoltage, leadPosition, leadVelocity);

    inputs.winchPosition = leadPosition.getValue();
    inputs.lowerLimit = lowerLimitSwitch.get();
    inputs.winchConnected = leadConnectedDebouncer.calculate(connectedStatus.isOK());
    inputs.winchVelocity = leadVelocity.getValue();
    inputs.winchCurrent = leadCurrent.getValue();
    inputs.winchAppliedVolts = leadVoltage.getValue();
  }
}

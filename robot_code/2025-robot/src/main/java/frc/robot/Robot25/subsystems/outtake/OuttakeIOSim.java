package frc.robot.Robot25.subsystems.outtake;

import static edu.wpi.first.units.Units.Amps;
import static edu.wpi.first.units.Units.RadiansPerSecond;
import static edu.wpi.first.units.Units.Seconds;
import static edu.wpi.first.units.Units.Volts;
import static frc.robot.Robot25.subsystems.outtake.OuttakeConstants.CURRENT_LIMIT;
import static frc.robot.Robot25.subsystems.outtake.OuttakeConstants.GEARING;

import edu.wpi.first.math.controller.SimpleMotorFeedforward;
import edu.wpi.first.math.system.plant.DCMotor;
import edu.wpi.first.math.system.plant.LinearSystemId;
import edu.wpi.first.units.measure.AngularVelocity;
import edu.wpi.first.units.measure.Voltage;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj.simulation.FlywheelSim;
import frc.lib.tunables.LoggedTunableBoolean;
import frc.robot.Robot25.subsystems.outtake.OuttakeConstants.Sim;
import org.ironmaple.simulation.motorsims.MapleMotorSim;
import org.ironmaple.simulation.motorsims.SimMotorConfigs;
import org.ironmaple.simulation.motorsims.SimulatedMotorController;

public class OuttakeIOSim implements OuttakeIO {
  private static final DCMotor outtakeGearbox = DCMotor.getKrakenX60(1);
  private final SimulatedMotorController.GenericMotorController outtakeMotorController;
  private final MapleMotorSim outtakeMotor;
  private boolean isClosedLoop = false;
  private Voltage outtakeAppliedVoltage = Volts.of(0);
  private final SimpleMotorFeedforward feedForwardController =
      new SimpleMotorFeedforward(Sim.kS, Sim.kV, Sim.kA);

  private final FlywheelSim outtakeSim = new FlywheelSim(
      LinearSystemId.createFlywheelSystem(outtakeGearbox, 0.1, GEARING), outtakeGearbox, 0.000015);

  LoggedTunableBoolean LoadSideSensor =
      new LoggedTunableBoolean("Tuning/Outtake/LoadSideSensor", false);
  LoggedTunableBoolean ScoreSideSensor =
      new LoggedTunableBoolean("Tuning/Outtake/ScoreSideSensor", false);

  public OuttakeIOSim() {
    outtakeMotor = new MapleMotorSim(
        new SimMotorConfigs(outtakeGearbox, GEARING, Sim.MOTOR_LOAD_MOI, Sim.FRICTION_VOLTAGE));
    outtakeMotorController =
        outtakeMotor.useSimpleDCMotorController().withCurrentLimit(CURRENT_LIMIT);
  }

  @Override
  public void setOpenLoop(Voltage output) {
    outtakeAppliedVoltage = output;
    isClosedLoop = false;
  }

  @Override
  public void updateInputs(OuttakeIOInputs inputs) {
    var angularVelocity = outtakeSim.getAngularVelocityRadPerSec();
    if (isClosedLoop) {
      var feedForwardVolts = feedForwardController.calculate(angularVelocity);
      outtakeAppliedVoltage = Volts.of(feedForwardVolts);
    }

    outtakeMotorController.requestVoltage(outtakeAppliedVoltage);
    outtakeSim.setInputVoltage(outtakeMotor.getAppliedVoltage().in(Volts));
    outtakeMotor.update(Seconds.of(TimedRobot.kDefaultPeriod));
    outtakeSim.update(TimedRobot.kDefaultPeriod);

    // Update motor inputs
    inputs.outtakeConnected = true;
    inputs.outtakeAppliedVolts = outtakeAppliedVoltage;
    inputs.outtakeCurrent = Amps.of(outtakeSim.getCurrentDrawAmps());
    inputs.outtakeVelocity = AngularVelocity.ofBaseUnits(angularVelocity, RadiansPerSecond);
    inputs.seesCoralAtInput = LoadSideSensor.get();
    inputs.seesCoralAtOutput = ScoreSideSensor.get();
  }
}

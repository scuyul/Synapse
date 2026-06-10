package frc.lib.tunables;

import edu.wpi.first.wpilibj2.command.CommandScheduler;
import frc.lib.functional.BooleanDoubleConsumer;
import java.util.function.DoubleConsumer;
import org.littletonrobotics.junction.networktables.LoggedNetworkNumber;

public class TunableDouble {
  private LoggedNetworkNumber value;

  /**
   * Creates a TunableDouble. It can be enabled and disabled (Use defaultValue)
   *
   * @param name
   * @param d
   * @param tunable
   */
  public TunableDouble(String name, double defaultValue, String tab) {
    this(name, defaultValue, true, tab);
  }

  public TunableDouble(String name, double defaultValue, String tab, DoubleConsumer onChange) {
    this(name, defaultValue, tab);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, String tab,
      BooleanDoubleConsumer onChange) {
    this(name, defaultValue, tab);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, boolean tunable, String tab) {
    value = new LoggedNetworkNumber("/Tuning/" + tab + "/" + name, defaultValue);

  }


  public TunableDouble(String name, double defaultValue, boolean tunable, String tab,
      DoubleConsumer onChange) {
    this(name, defaultValue, tunable, tab);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, boolean tunable, DoubleConsumer onChange) {
    this(name, defaultValue, tunable);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, boolean tunable, String tab,
      BooleanDoubleConsumer onChange) {
    this(name, defaultValue, tunable, tab);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, boolean tunable,
      BooleanDoubleConsumer onChange) {
    this(name, defaultValue, tunable);
    addChangeListener(onChange);
  }

  public TunableDouble(String name, double defaultValue, boolean tunable) {
    this(name, defaultValue, tunable, "Tunables");
  }


  /**
   * @return Value as a double
   */
  public double getValue() {
    return value.get();
  }

  public void addChangeListener(DoubleConsumer onChange) {
    addChangeListener((isInit, value) -> onChange.accept(value));
  }

  public void addChangeListener(BooleanDoubleConsumer onChange) {
    onChange.accept(true, getValue());
    CommandScheduler.getInstance().getDefaultButtonLoop().bind(new Runnable() {
      private double oldValue = getValue();

      @Override
      public void run() {
        double newValue = getValue();

        if (oldValue != newValue) {
          onChange.accept(false, newValue);
          oldValue = newValue;
        }
      }
    });
  }
}

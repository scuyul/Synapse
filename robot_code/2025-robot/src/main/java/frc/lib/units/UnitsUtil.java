package frc.lib.units;

import static edu.wpi.first.units.Units.Meters;
import static edu.wpi.first.units.Units.MetersPerSecond;
import static edu.wpi.first.units.Units.Radian;
import static edu.wpi.first.units.Units.Rotations;
import static edu.wpi.first.units.Units.RotationsPerSecond;

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.units.ImmutableMeasure;
import edu.wpi.first.units.Measure;
import edu.wpi.first.units.Unit;
import edu.wpi.first.units.measure.*;

public final class UnitsUtil {
  private UnitsUtil() {
    // hi - dont make instances
  }

  // /* Standard unit of measurement for frequency, hertz */
  // public static final Frequency hertz = new Frequency(1, "Hertz", "Hz");
  // /* Standard unit of measurement for frequency, mega hertz */
  // public static final Frequency megaHertz = derive(hertz).aggregate(1000).named("Mega
  // Hertz").symbol("Mhz")
  // .make();
  // /* Standard unit of measurement for frequency, giga hertz */
  // public static final Frequency gigaHertz = derive(megaHertz).aggregate(1000).named("Giga
  // Hertz").symbol("Ghz")
  // .make();

  public static <U extends Unit> Measure<U> abs(Measure<U> measure) {
    return ImmutableMeasure.ofBaseUnits(measure.abs(measure.unit()), measure.unit());
  }

  public static final Distance distanceForWheel(Distance wheelDiameter, Angle rotations) {
    var distance = Math.PI * wheelDiameter.in(Meters) * rotations.in(Rotations);
    return Meters.of(distance);
  }

  public static final LinearVelocity velocityForWheel(Distance wheelDiameter,
      AngularVelocity rotations) {
    var distance = Math.PI * wheelDiameter.in(Meters) * rotations.in(RotationsPerSecond);
    return MetersPerSecond.of(distance);
  }

  public static <U extends Unit> Measure<U> inputModulus(Measure<U> value, Measure<U> min,
      Measure<U> max) {
    U unit = value.unit();
    double dvalue = value.in(unit);
    double dmin = value.in(unit);
    double dmax = value.in(unit);
    return ImmutableMeasure.ofRelativeUnits(MathUtil.inputModulus(dvalue, dmin, dmax), unit);
  }

  public static final Angle angleModulus(Angle value) {
    return Radian.of(MathUtil.angleModulus(value.in(Radian)));
  }
}

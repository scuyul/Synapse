package frc.lib;

import edu.wpi.first.networktables.GenericEntry;
import edu.wpi.first.wpilibj.shuffleboard.BuiltInWidgets;
import edu.wpi.first.wpilibj.shuffleboard.Shuffleboard;

public class GraphWrapper {
  private String name;
  private GenericEntry graph;

  public GraphWrapper(String name, String tab) {
    this.name = name;
    this.graph = Shuffleboard.getTab(tab).add(name, 0).withWidget(BuiltInWidgets.kGraph).getEntry();
  }

  public void setDouble(double value) {
    if (graph != null)
      graph.setDouble(value);
  }
}

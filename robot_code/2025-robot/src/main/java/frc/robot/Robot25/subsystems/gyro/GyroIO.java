// Copyright 2021-2025 FRC 6328
// http://github.com/Mechanical-Advantage
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// version 3 as published by the Free Software Foundation or
// available in the root directory of this project.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

package frc.robot.Robot25.subsystems.gyro;

import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Rotation3d;

import org.littletonrobotics.junction.AutoLog;

public interface GyroIO {
  @AutoLog
  public static class GyroIOInputs {
    public boolean connected = false;

    public Rotation3d gyroRotation = Rotation3d.kZero;

    public Rotation2d yawPosition = new Rotation2d();
    public double yawVelocityRadPerSec = 0.0;

    public Rotation2d rollPosition = new Rotation2d();
    public double rollVelocityRadPerSec = 0.0;

    public Rotation2d pitchPosition = new Rotation2d();
    public double pitchVelocityRadPerSec = 0.0;

    public double[] odometryYawTimestamps = new double[] {};
    public Rotation2d[] odometryYawPositions = new Rotation2d[] {};
    public Rotation3d[] odometryRotation3d = new Rotation3d[] {};
  }

  public default void updateInputs(GyroIOInputs inputs) {}
}

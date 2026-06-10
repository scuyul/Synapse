"""Integration helpers for copied FRC robot-code projects."""

from __future__ import annotations

from dataclasses import dataclass, replace
import ast
import math
from pathlib import Path
import re
from typing import Any

from reefscape_rl.constants import ROBOT_RADIUS_M


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_2025_ROBOT_ROOT = REPO_ROOT / "robot_code" / "2025-robot"
DRIVE_CONSTANTS_PATH = "src/main/java/frc/robot/Robot25/subsystems/drive/DriveConstants.java"
INCH_TO_METER = 0.0254


@dataclass(frozen=True, slots=True)
class RobotProfile:
    name: str
    source_root: Path
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    drive_base_radius_m: float
    wheel_radius_m: float
    robot_radius_m: float
    robot_mass_kg: float
    robot_moi: float
    drive_gear_ratio: float
    module_locations_m: dict[str, tuple[float, float]]


def load_robot_profile(name: str = "sim", root: Path | None = None) -> RobotProfile | None:
    """Load an optional robot profile.

    ``sim`` returns ``None`` so callers can keep the built-in simulator defaults.
    ``2025-robot`` parses the copied Java robot constants.
    """

    if name in {"", "sim", "default"}:
        return None
    if name != "2025-robot":
        raise ValueError(f"unknown robot profile: {name}")
    return load_2025_robot_profile(root or DEFAULT_2025_ROBOT_ROOT)


def load_2025_robot_profile(root: Path = DEFAULT_2025_ROBOT_ROOT) -> RobotProfile:
    constants_path = root / DRIVE_CONSTANTS_PATH
    if not constants_path.exists():
        raise FileNotFoundError(f"robot drive constants not found: {constants_path}")
    text = constants_path.read_text(encoding="utf-8")

    speed = _extract_unit_value(text, "kSpeedAt12Volts", "MetersPerSecond")
    wheel_radius = _extract_unit_value(text, "kWheelRadius", "Inches") * INCH_TO_METER
    module_locations = {
        "front_left": (
            _extract_unit_value(text, "kFrontLeftXPos", "Inches") * INCH_TO_METER,
            _extract_unit_value(text, "kFrontLeftYPos", "Inches") * INCH_TO_METER,
        ),
        "front_right": (
            _extract_unit_value(text, "kFrontRightXPos", "Inches") * INCH_TO_METER,
            _extract_unit_value(text, "kFrontRightYPos", "Inches") * INCH_TO_METER,
        ),
        "back_left": (
            _extract_unit_value(text, "kBackLeftXPos", "Inches") * INCH_TO_METER,
            _extract_unit_value(text, "kBackLeftYPos", "Inches") * INCH_TO_METER,
        ),
        "back_right": (
            _extract_unit_value(text, "kBackRightXPos", "Inches") * INCH_TO_METER,
            _extract_unit_value(text, "kBackRightYPos", "Inches") * INCH_TO_METER,
        ),
    }
    drive_base_radius = max(math.hypot(x, y) for x, y in module_locations.values())
    max_angular_speed = speed / drive_base_radius

    return RobotProfile(
        name="2025-robot",
        source_root=root,
        max_linear_speed_mps=speed,
        max_angular_speed_radps=max_angular_speed,
        drive_base_radius_m=drive_base_radius,
        wheel_radius_m=wheel_radius,
        robot_radius_m=max(ROBOT_RADIUS_M, drive_base_radius + 0.10),
        robot_mass_kg=_extract_double(text, "ROBOT_MASS_KG"),
        robot_moi=_extract_double(text, "ROBOT_MOI"),
        drive_gear_ratio=_extract_double(text, "kDriveGearRatio"),
        module_locations_m=module_locations,
    )


def apply_profile_to_env_config(config: Any, profile: RobotProfile | None) -> Any:
    if profile is None:
        return config
    return replace(
        config,
        max_linear_speed_mps=profile.max_linear_speed_mps,
        max_angular_speed_radps=profile.max_angular_speed_radps,
        robot_radius_m=profile.robot_radius_m,
        other_robot_radius_m=profile.robot_radius_m,
    )


def _extract_double(text: str, name: str) -> float:
    match = re.search(rf"\b{name}\s*=\s*([^;]+);", text)
    if not match:
        raise ValueError(f"missing Java constant: {name}")
    return _safe_eval_number(match.group(1))


def _extract_unit_value(text: str, name: str, unit: str) -> float:
    pattern = rf"\b{name}\s*=\s*{unit}\.of\(([^)]+)\)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing Java {unit} constant: {name}")
    return _safe_eval_number(match.group(1))


def _safe_eval_number(expression: str) -> float:
    tree = ast.parse(expression.strip(), mode="eval")
    return float(_eval_ast(tree.body))


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError(f"unsupported numeric expression: {ast.dump(node)}")

"""Optional Xbox controller support for manually driving the defense robot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class XboxController:
    joystick: object
    deadband: float = 0.08

    @classmethod
    def open_first(cls, *, deadband: float = 0.08) -> "XboxController":
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pygame is required for Xbox controller support.") from exc

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No controller found. Connect an Xbox controller and try again.")
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Xbox defense controller: {joystick.get_name()}")
        return cls(joystick=joystick, deadband=deadband)

    def command(self) -> tuple[float, float, float]:
        import pygame

        pygame.event.pump()
        left_x = _apply_deadband(float(self.joystick.get_axis(0)), self.deadband)
        left_y = _apply_deadband(float(self.joystick.get_axis(1)), self.deadband)
        right_x = 0.0
        if self.joystick.get_numaxes() > 2:
            right_x = _apply_deadband(float(self.joystick.get_axis(2)), self.deadband)
        return left_x, -left_y, right_x

    def close(self) -> None:
        try:
            import pygame

            self.joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass


def _apply_deadband(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / max(1e-6, 1.0 - deadband)
    return scaled if value > 0.0 else -scaled

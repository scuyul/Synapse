"""AdvantageScope visualization callback for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from reefscape_rl.action_adapter import ResidualHeuristicActionAdapter
from reefscape_rl.env import ReefscapeEnv, ReefscapeEnvConfig
from reefscape_rl.nt_publisher import AdvantageScopeNtPublisher
from reefscape_rl.visualizer_snapshot import build_visualizer_snapshot


@dataclass(slots=True)
class TrainingVisualizationConfig:
    port: int = 5810
    every_steps: int = 512
    preview_steps: int = 25
    seed: int = 2026
    open_app: bool = False


class AdvantageScopeTrainingCallback(BaseCallback):
    """Streams periodic policy rollouts while PPO trains."""

    def __init__(self, config: TrainingVisualizationConfig):
        super().__init__()
        self.config = config
        self.publisher: AdvantageScopeNtPublisher | None = None
        self.preview_env = ReefscapeEnv(
            ReefscapeEnvConfig(
                randomize_start=False,
                auto_mechanisms=False,
                randomize_other_robot_start=True,
                randomize_other_robot_behavior=True,
            )
        )
        self.action_adapter = ResidualHeuristicActionAdapter()
        self.preview_obs: np.ndarray | None = None
        self.preview_episode = 0
        self.preview_return = 0.0

    def _on_training_start(self) -> None:
        self.publisher = AdvantageScopeNtPublisher.start_server(port=self.config.port)
        if self.config.open_app:
            _try_open_advantagescope()
        obs, _ = self.preview_env.reset(seed=self.config.seed)
        self.preview_obs = np.asarray(obs, dtype=np.float32)
        self.publisher.publish(self.preview_env)
        self.publisher.publish_training(
            training_step=0,
            preview_episode_return=0.0,
            preview_episode=self.preview_episode,
        )
        print(f"AdvantageScope training stream started on 127.0.0.1:{self.config.port}")

    def _on_step(self) -> bool:
        if self.publisher is None or self.preview_obs is None:
            return True
        if self.num_timesteps % self.config.every_steps != 0:
            return True

        for _ in range(self.config.preview_steps):
            self.publisher.apply_tunables(self.preview_env)
            action, _ = self.model.predict(self.preview_obs, deterministic=True)
            sim_action = self.action_adapter.adapt(self.preview_env, action)
            obs, reward, terminated, truncated, _ = self.preview_env.step(sim_action)
            self.preview_obs = np.asarray(obs, dtype=np.float32)
            self.preview_return += float(reward)
            self.publisher.publish(self.preview_env, reward=float(reward))
            self.publisher.publish_training(
                training_step=self.num_timesteps,
                preview_episode_return=self.preview_return,
                preview_episode=self.preview_episode,
            )
            if terminated or truncated:
                self.preview_episode += 1
                obs, _ = self.preview_env.reset(seed=self.config.seed + self.preview_episode)
                self.preview_obs = np.asarray(obs, dtype=np.float32)
                self.preview_return = 0.0
                self.publisher.publish(self.preview_env)
                break

        return True


@dataclass(slots=True)
class CustomUiTrainingConfig:
    port: int = 8775
    state_path: Path = Path("logs/reefscape_visualizer_state.json")
    every_steps: int = 512
    preview_steps: int = 25
    seed: int = 2026
    open_app: bool = False


class CustomUiTrainingCallback(BaseCallback):
    """Streams periodic policy rollouts to the native REEFSCAPE desktop visualizer."""

    def __init__(self, config: CustomUiTrainingConfig):
        super().__init__()
        self.config = config
        self.preview_env = ReefscapeEnv(
            ReefscapeEnvConfig(
                randomize_start=False,
                auto_mechanisms=False,
                randomize_other_robot_start=True,
                randomize_other_robot_behavior=True,
            )
        )
        self.action_adapter = ResidualHeuristicActionAdapter()
        self.preview_obs: np.ndarray | None = None
        self.preview_episode = 0
        self.preview_return = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.75]
        self.last_reward = 0.0

    def _on_training_start(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        obs, _ = self.preview_env.reset(seed=self.config.seed)
        self.preview_obs = np.asarray(obs, dtype=np.float32)
        self._write_snapshot()
        if self.config.open_app:
            _try_open_desktop_app()
        print(f"Native REEFSCAPE visualizer state started at {self.config.state_path}")

    def _on_step(self) -> bool:
        if self.preview_obs is None:
            return True
        if self.num_timesteps % self.config.every_steps != 0:
            return True

        for _ in range(self.config.preview_steps):
            action, _ = self.model.predict(self.preview_obs, deterministic=True)
            sim_action = self.action_adapter.adapt(self.preview_env, action)
            obs, reward, terminated, truncated, _ = self.preview_env.step(sim_action)
            self.preview_obs = np.asarray(obs, dtype=np.float32)
            self.preview_return += float(reward)
            self.last_action = [float(value) for value in sim_action]
            self.last_reward = float(reward)
            self._write_snapshot()
            if terminated or truncated:
                self.preview_episode += 1
                obs, _ = self.preview_env.reset(seed=self.config.seed + self.preview_episode)
                self.preview_obs = np.asarray(obs, dtype=np.float32)
                self.preview_return = 0.0
                self._write_snapshot()
                break
        return True

    def _write_snapshot(self) -> None:
        payload = build_visualizer_snapshot(
            self.preview_env,
            action=self.last_action,
            reward=self.last_reward,
            policy_name="training model",
            running=True,
        )
        payload["training"] = {
            "step": int(self.num_timesteps),
            "previewEpisode": int(self.preview_episode),
            "previewReturn": float(self.preview_return),
        }
        tmp_path = self.config.state_path.with_name(
            f"{self.config.state_path.stem}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(8):
            try:
                tmp_path.replace(self.config.state_path)
                return
            except PermissionError:
                if attempt == 7:
                    break
                time.sleep(0.025 * (attempt + 1))

        try:
            self.config.state_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"Warning: could not update custom visualizer state: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

def _try_open_advantagescope() -> None:
    executable = _find_advantagescope_executable()
    if executable is None:
        print(
            "AdvantageScope preview is running, but AdvantageScope could not be opened automatically."
        )
        print("Open AdvantageScope manually and connect NetworkTables to 127.0.0.1.")
        return
    try:
        subprocess.Popen([str(executable)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Opened AdvantageScope: {executable}")
    except OSError as exc:
        print(f"Could not open AdvantageScope automatically: {exc}")


def _find_advantagescope_executable() -> Path | None:
    env_path = os.environ.get("ADVANTAGESCOPE_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    which_path = shutil.which("advantagescope") or shutil.which("AdvantageScope")
    if which_path:
        candidates.append(Path(which_path))
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
    for root in (local_app_data, program_files, program_files_x86):
        if not root:
            continue
        candidates.extend(
            [
                Path(root) / "Programs" / "AdvantageScope" / "AdvantageScope.exe",
                Path(root) / "AdvantageScope" / "AdvantageScope.exe",
            ]
        )
    candidates.extend(
        [
            Path("C:/Program Files/AdvantageScope/AdvantageScope.exe"),
            Path("C:/Program Files (x86)/AdvantageScope/AdvantageScope.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _try_open_desktop_app() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "ReefscapeRL.exe",
        repo_root / "builds" / "ReefscapeRL" / "ReefscapeRL.exe",
        repo_root / "builds" / "reefscape-app.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                subprocess.Popen([str(candidate)], cwd=repo_root)
                print(f"Opened REEFSCAPE desktop app: {candidate}")
                return
            except OSError as exc:
                print(f"Could not open REEFSCAPE desktop app: {exc}")
                return
    print("Native visualizer state is active. Open ReefscapeRL.exe to view it.")

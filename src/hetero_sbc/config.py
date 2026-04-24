from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ScenarioConfig:
    name: str
    positions: np.ndarray
    velocities: np.ndarray
    goals: np.ndarray
    accel_limits: np.ndarray
    speed_limits: np.ndarray
    radii: np.ndarray
    gamma: np.ndarray
    dt: float = 0.05
    steps: int = 600
    run_until_complete: bool = False
    kp: float = 1.0
    kd: float = 1.8
    safety_buffer: float = 0.0
    estimate_floor: float | None = None
    estimate_gain: float = 3.0


@dataclass
class ExperimentResult:
    name: str
    controller: str
    summary: dict
    time: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    controls: np.ndarray
    nominal_controls: np.ndarray
    clearance_history: np.ndarray
    cbf_history: np.ndarray
    qp_times_ms: np.ndarray
    estimation_history: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)

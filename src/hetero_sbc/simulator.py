from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .barriers import cbf_value, pairwise_clearance, pairwise_safe_distance
from .config import ExperimentResult, ScenarioConfig
from .controllers import (
    ControlStep,
    heterogeneous_barrier_controller,
    nominal_controller,
    symmetric_barrier_controller,
    uncertain_heterogeneous_barrier_controller,
    update_alpha_estimates,
)


ControllerFn = Callable[..., ControlStep]


def _controller_lookup(controller_name: str) -> ControllerFn:
    if controller_name == "nominal":
        return nominal_controller
    if controller_name == "heterogeneous_barrier":
        return heterogeneous_barrier_controller
    if controller_name == "symmetric_barrier":
        return symmetric_barrier_controller
    if controller_name == "uncertain_heterogeneous_barrier":
        return uncertain_heterogeneous_barrier_controller
    raise ValueError(f"Unknown controller: {controller_name}")


def _compute_pairwise_cbf(
    positions: np.ndarray,
    velocities: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    safety_buffer: float,
) -> np.ndarray:
    n_agents = positions.shape[0]
    values = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            delta_p = positions[i] - positions[j]
            delta_v = velocities[i] - velocities[j]
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            values.append(cbf_value(delta_p, delta_v, accel_limits[i], accel_limits[j], safe_distance))
    return np.asarray(values, dtype=float)


def _step_dynamics(
    positions: np.ndarray,
    velocities: np.ndarray,
    controls: np.ndarray,
    speed_limits: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    new_velocities = velocities + dt * controls
    speeds = np.linalg.norm(new_velocities, axis=1)
    for i, speed in enumerate(speeds):
        if speed > speed_limits[i]:
            new_velocities[i] *= speed_limits[i] / speed
    new_positions = positions + dt * new_velocities
    return new_positions, new_velocities


def _task_completion_time(
    positions: np.ndarray,
    goals: np.ndarray,
    threshold: float = 0.15,
) -> float | None:
    for step in range(positions.shape[0]):
        if np.all(np.linalg.norm(positions[step] - goals, axis=1) <= threshold):
            return float(step)
    return None


def simulate_scenario(config: ScenarioConfig, controller_name: str) -> ExperimentResult:
    controller = _controller_lookup(controller_name)
    n_agents = config.positions.shape[0]

    positions = np.zeros((config.steps + 1, n_agents, 2), dtype=float)
    velocities = np.zeros((config.steps + 1, n_agents, 2), dtype=float)
    controls = np.zeros((config.steps, n_agents, 2), dtype=float)
    nominal_controls = np.zeros((config.steps, n_agents, 2), dtype=float)
    qp_times_ms = np.zeros((config.steps, n_agents), dtype=float)
    clearance_history = np.zeros((config.steps + 1,), dtype=float)
    cbf_history = np.zeros((config.steps + 1,), dtype=float)
    estimation_history = None

    positions[0] = config.positions
    velocities[0] = config.velocities
    clearance_history[0] = pairwise_clearance(positions[0], config.radii, config.safety_buffer).min()
    cbf_history[0] = _compute_pairwise_cbf(
        positions[0], velocities[0], config.accel_limits, config.radii, config.safety_buffer
    ).min()

    alpha_estimates = None
    if controller_name == "uncertain_heterogeneous_barrier":
        floor = config.estimate_floor if config.estimate_floor is not None else float(np.min(config.accel_limits))
        alpha_estimates = np.full((n_agents, n_agents), floor, dtype=float)
        np.fill_diagonal(alpha_estimates, config.accel_limits)
        estimation_history = np.zeros((config.steps + 1, n_agents, n_agents), dtype=float)
        estimation_history[0] = alpha_estimates

    last_step = config.steps
    for step in range(config.steps):
        kwargs = dict(
            positions=positions[step],
            velocities=velocities[step],
            goals=config.goals,
            accel_limits=config.accel_limits,
            kp=config.kp,
            kd=config.kd,
        )
        if controller_name != "nominal":
            kwargs.update(
                radii=config.radii,
                gamma=config.gamma,
                safety_buffer=config.safety_buffer,
            )
        if controller_name == "uncertain_heterogeneous_barrier":
            kwargs["alpha_estimates"] = alpha_estimates

        step_result = controller(**kwargs)
        controls[step] = step_result.control
        nominal_controls[step] = step_result.nominal
        qp_times_ms[step] = step_result.qp_times_ms
        positions[step + 1], velocities[step + 1] = _step_dynamics(
            positions[step],
            velocities[step],
            step_result.control,
            config.speed_limits,
            config.dt,
        )
        clearance_history[step + 1] = pairwise_clearance(
            positions[step + 1], config.radii, config.safety_buffer
        ).min()
        cbf_history[step + 1] = _compute_pairwise_cbf(
            positions[step + 1],
            velocities[step + 1],
            config.accel_limits,
            config.radii,
            config.safety_buffer,
        ).min()
        if controller_name == "uncertain_heterogeneous_barrier":
            alpha_estimates = update_alpha_estimates(
                alpha_estimates,
                step_result.control,
                config.accel_limits,
                config.dt,
                config.estimate_gain,
            )
            estimation_history[step + 1] = alpha_estimates
        if np.all(np.linalg.norm(positions[step + 1] - config.goals, axis=1) <= 0.15) and np.all(
            np.linalg.norm(velocities[step + 1], axis=1) <= 0.05
        ):
            last_step = step + 1
            break

    positions = positions[: last_step + 1]
    velocities = velocities[: last_step + 1]
    controls = controls[:last_step]
    nominal_controls = nominal_controls[:last_step]
    qp_times_ms = qp_times_ms[:last_step]
    clearance_history = clearance_history[: last_step + 1]
    cbf_history = cbf_history[: last_step + 1]
    if estimation_history is not None:
        estimation_history = estimation_history[: last_step + 1]

    completion_step = _task_completion_time(positions, config.goals)
    control_deviation = np.linalg.norm(controls - nominal_controls, axis=2)
    summary = {
        "min_clearance": float(clearance_history.min()),
        "min_cbf": float(cbf_history.min()),
        "collision": bool(clearance_history.min() < 0.0),
        "mean_qp_ms": float(qp_times_ms.mean()),
        "p95_qp_ms": float(np.percentile(qp_times_ms, 95)),
        "max_qp_ms": float(qp_times_ms.max()),
        "completion_step": completion_step,
        "mean_goal_error": float(np.linalg.norm(positions[-1] - config.goals, axis=1).mean()),
        "mean_control_deviation": float(control_deviation.mean()),
        "per_agent_control_deviation": control_deviation.mean(axis=0).tolist(),
    }
    if estimation_history is not None:
        summary["final_mean_alpha_estimate_error"] = float(
            np.mean(np.abs(estimation_history[-1] - config.accel_limits.reshape(1, -1)))
        )

    return ExperimentResult(
        name=config.name,
        controller=controller_name,
        summary=summary,
        time=np.arange(config.steps + 1, dtype=float) * config.dt,
        positions=positions,
        velocities=velocities,
        controls=controls,
        nominal_controls=nominal_controls,
        clearance_history=clearance_history,
        cbf_history=cbf_history,
        qp_times_ms=qp_times_ms,
        estimation_history=estimation_history,
        metadata={
            "accel_limits": config.accel_limits.tolist(),
            "speed_limits": config.speed_limits.tolist(),
            "gamma": config.gamma.tolist(),
            "radii": config.radii.tolist(),
            "safety_buffer": config.safety_buffer,
        },
    )

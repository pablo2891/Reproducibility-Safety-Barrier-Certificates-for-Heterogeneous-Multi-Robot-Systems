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
    unclipped_velocities = velocities + dt * controls
    new_velocities = unclipped_velocities.copy()
    speeds = np.linalg.norm(unclipped_velocities, axis=1)
    for i, speed in enumerate(speeds):
        if speed > speed_limits[i]:
            new_velocities[i] *= speed_limits[i] / speed
    # Use the average velocity over the step so the discrete dynamics
    # better match the double-integrator model under bounded acceleration.
    new_positions = positions + 0.5 * dt * (velocities + new_velocities)
    return new_positions, new_velocities


def _clip_controls(controls: np.ndarray, accel_limits: np.ndarray) -> np.ndarray:
    clipped = controls.copy()
    for i, limit in enumerate(accel_limits):
        clipped[i] = np.clip(clipped[i], -limit, limit)
    return clipped


def _enforce_discrete_safety(
    positions: np.ndarray,
    velocities: np.ndarray,
    controls: np.ndarray,
    accel_limits: np.ndarray,
    speed_limits: np.ndarray,
    radii: np.ndarray,
    safety_buffer: float,
    dt: float,
    passes: int = 8,
) -> np.ndarray:
    adjusted = _clip_controls(controls, accel_limits)
    for _ in range(passes):
        predicted_positions, _ = _step_dynamics(positions, velocities, adjusted, speed_limits, dt)
        updated = False
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
                delta = predicted_positions[i] - predicted_positions[j]
                distance = float(np.linalg.norm(delta))
                if distance >= safe_distance:
                    continue
                if distance <= 1e-9:
                    delta = positions[i] - positions[j]
                    distance = max(float(np.linalg.norm(delta)), 1e-9)
                direction = delta / distance
                deficit = safe_distance - distance + 5e-3
                correction = deficit / max(dt * dt, 1e-9)
                adjusted[i] += correction * direction
                adjusted[j] -= correction * direction
                updated = True
        if not updated:
            break
        adjusted = _clip_controls(adjusted, accel_limits)
    return adjusted


def _task_completion_time(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    goal_tolerance: float,
    velocity_tolerance: float,
) -> float | None:
    for step in range(positions.shape[0]):
        if _all_goals_reached(positions[step], velocities[step], goals, goal_tolerance, velocity_tolerance):
            return float(step)
    return None


def _all_goals_reached(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    goal_tolerance: float,
    velocity_tolerance: float,
) -> bool:
    return bool(
        np.all(np.linalg.norm(positions - goals, axis=1) <= goal_tolerance)
        and np.all(np.linalg.norm(velocities, axis=1) <= velocity_tolerance)
    )


def _minimum_pair_distance(positions: np.ndarray) -> float:
    n_agents = positions.shape[0]
    min_distance = float("inf")
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            min_distance = min(min_distance, float(np.linalg.norm(positions[i] - positions[j])))
    return 0.0 if min_distance == float("inf") else min_distance


def simulate_scenario(config: ScenarioConfig, controller_name: str) -> ExperimentResult:
    controller = _controller_lookup(controller_name)
    n_agents = config.positions.shape[0]

    positions_history: list[np.ndarray] = [config.positions.copy()]
    velocities_history: list[np.ndarray] = [config.velocities.copy()]
    controls_history: list[np.ndarray] = []
    nominal_controls_history: list[np.ndarray] = []
    qp_times_history: list[np.ndarray] = []
    clearance_history: list[float] = [
        float(pairwise_clearance(config.positions, config.radii, config.safety_buffer).min())
    ]
    cbf_history: list[float] = [
        float(
            _compute_pairwise_cbf(
                config.positions,
                config.velocities,
                config.accel_limits,
                config.radii,
                config.safety_buffer,
            ).min()
        )
    ]
    estimation_history = None
    termination_reason = "step_limit"

    alpha_estimates = None
    if controller_name == "uncertain_heterogeneous_barrier":
        floor = config.estimate_floor if config.estimate_floor is not None else float(np.min(config.accel_limits))
        alpha_estimates = np.full((n_agents, n_agents), floor, dtype=float)
        np.fill_diagonal(alpha_estimates, config.accel_limits)
        estimation_history = [alpha_estimates.copy()]

    step = 0
    while True:
        kwargs = dict(
            positions=positions_history[-1],
            velocities=velocities_history[-1],
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
        applied_control = step_result.control
        if controller_name != "nominal":
            applied_control = _enforce_discrete_safety(
                positions=positions_history[-1],
                velocities=velocities_history[-1],
                controls=step_result.control,
                accel_limits=config.accel_limits,
                speed_limits=config.speed_limits,
                radii=config.radii,
                safety_buffer=config.safety_buffer,
                dt=config.dt,
            )
        controls_history.append(applied_control.copy())
        nominal_controls_history.append(step_result.nominal.copy())
        qp_times_history.append(step_result.qp_times_ms.copy())
        next_positions, next_velocities = _step_dynamics(
            positions_history[-1],
            velocities_history[-1],
            applied_control,
            config.speed_limits,
            config.dt,
        )
        positions_history.append(next_positions)
        velocities_history.append(next_velocities)
        clearance_history.append(float(pairwise_clearance(next_positions, config.radii, config.safety_buffer).min()))
        cbf_history.append(
            float(
                _compute_pairwise_cbf(
                    next_positions,
                    next_velocities,
                    config.accel_limits,
                    config.radii,
                    config.safety_buffer,
                ).min()
            )
        )
        if clearance_history[-1] < 0.0:
            termination_reason = "collision"
            break
        if controller_name == "uncertain_heterogeneous_barrier":
            alpha_estimates = update_alpha_estimates(
                alpha_estimates,
                step_result.control,
                config.accel_limits,
                config.dt,
                config.estimate_gain,
            )
            estimation_history.append(alpha_estimates.copy())
        if _all_goals_reached(
            next_positions,
            next_velocities,
            config.goals,
            config.goal_tolerance,
            config.velocity_tolerance,
        ):
            termination_reason = "all_goals_reached"
            break
        step += 1
        if step >= config.steps:
            break

    positions = np.asarray(positions_history, dtype=float)
    velocities = np.asarray(velocities_history, dtype=float)
    controls = np.asarray(controls_history, dtype=float)
    nominal_controls = np.asarray(nominal_controls_history, dtype=float)
    qp_times_ms = np.asarray(qp_times_history, dtype=float)
    clearance_history_array = np.asarray(clearance_history, dtype=float)
    cbf_history_array = np.asarray(cbf_history, dtype=float)
    if estimation_history is not None:
        estimation_history = np.asarray(estimation_history, dtype=float)
    last_step = positions.shape[0] - 1

    completion_step = _task_completion_time(
        positions,
        velocities,
        config.goals,
        config.goal_tolerance,
        config.velocity_tolerance,
    )
    all_goals_reached = completion_step is not None
    control_deviation = np.linalg.norm(controls - nominal_controls, axis=2)
    summary = {
        "min_clearance": float(clearance_history_array.min()),
        "min_pair_distance": float(
            min(_minimum_pair_distance(position_step) for position_step in positions)
        ),
        "min_cbf": float(cbf_history_array.min()),
        "collision": bool(clearance_history_array.min() < 0.0),
        "all_goals_reached": all_goals_reached,
        "mean_qp_ms": float(qp_times_ms.mean()),
        "p95_qp_ms": float(np.percentile(qp_times_ms, 95)),
        "max_qp_ms": float(qp_times_ms.max()),
        "completion_step": completion_step,
        "termination_reason": termination_reason,
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
        time=np.arange(last_step + 1, dtype=float) * config.dt,
        positions=positions,
        velocities=velocities,
        controls=controls,
        nominal_controls=nominal_controls,
        clearance_history=clearance_history_array,
        cbf_history=cbf_history_array,
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

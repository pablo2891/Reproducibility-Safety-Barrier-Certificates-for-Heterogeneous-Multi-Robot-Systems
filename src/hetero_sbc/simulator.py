from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .barriers import cbf_value, pairwise_clearance, pairwise_safe_distance
from .config import ExperimentResult, ScenarioConfig
from .controllers import (
    ControlStep,
    centralized_heterogeneous_barrier_controller,
    centralized_uncertain_heterogeneous_barrier_controller,
    decentralized_heterogeneous_barrier_controller,
    decentralized_uncertain_heterogeneous_barrier_controller,
    heterogeneous_barrier_controller,
    nominal_controller,
    symmetric_barrier_controller,
    uncertain_heterogeneous_barrier_controller,
    update_alpha_estimates,
)


ControllerFn = Callable[..., ControlStep]
FAILURE_WINDOW_RADIUS = 5


def _controller_lookup(controller_name: str) -> ControllerFn:
    if controller_name == "nominal":
        return nominal_controller
    if controller_name == "heterogeneous_barrier":
        return heterogeneous_barrier_controller
    if controller_name == "decentralized_heterogeneous_barrier":
        return decentralized_heterogeneous_barrier_controller
    if controller_name == "centralized_heterogeneous_barrier":
        return centralized_heterogeneous_barrier_controller
    if controller_name == "symmetric_barrier":
        return symmetric_barrier_controller
    if controller_name == "uncertain_heterogeneous_barrier":
        return uncertain_heterogeneous_barrier_controller
    if controller_name == "decentralized_uncertain_heterogeneous_barrier":
        return decentralized_uncertain_heterogeneous_barrier_controller
    if controller_name == "centralized_uncertain_heterogeneous_barrier":
        return centralized_uncertain_heterogeneous_barrier_controller
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


def _pair_indices(n_agents: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{pair[0]}-{pair[1]}"


def _compute_pairwise_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    safety_buffer: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_agents = positions.shape[0]
    clearances = []
    cbf_values = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            delta_p = positions[i] - positions[j]
            delta_v = velocities[i] - velocities[j]
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            clearances.append(float(np.linalg.norm(delta_p) - safe_distance))
            cbf_values.append(cbf_value(delta_p, delta_v, accel_limits[i], accel_limits[j], safe_distance))
    return np.asarray(clearances, dtype=float), np.asarray(cbf_values, dtype=float)


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


def _agent_min_clearances(
    positions: np.ndarray,
    radii: np.ndarray,
    safety_buffer: float,
) -> np.ndarray:
    n_agents = positions.shape[0]
    agent_clearances = np.full(n_agents, np.inf, dtype=float)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            clearance = float(np.linalg.norm(positions[i] - positions[j]) - safe_distance)
            agent_clearances[i] = min(agent_clearances[i], clearance)
            agent_clearances[j] = min(agent_clearances[j], clearance)
    agent_clearances[~np.isfinite(agent_clearances)] = 0.0
    return agent_clearances


def _window_series(values: np.ndarray, center_step: int, radius: int = FAILURE_WINDOW_RADIUS) -> dict:
    start = max(0, center_step - radius)
    end = min(values.shape[0] - 1, center_step + radius)
    steps = list(range(start, end + 1))
    return {"steps": steps, "values": values[start : end + 1].astype(float).tolist()}


def _attach_pair_event_diagnostics(
    event: dict | None,
    pair_index: int | None,
    pairwise_cbf_history: np.ndarray,
    pairwise_clearance_history: np.ndarray,
    margin_step: int,
) -> dict | None:
    if event is None or pair_index is None:
        return event
    event_step = int(event["step"])
    event["pair_h_ij_at_event"] = float(pairwise_cbf_history[event_step, pair_index])
    event["pair_clearance_at_event"] = float(pairwise_clearance_history[event_step, pair_index])
    event["pair_h_margin_to_violation"] = float(pairwise_cbf_history[margin_step, pair_index])
    event["pair_clearance_margin_to_violation"] = float(pairwise_clearance_history[margin_step, pair_index])
    event["pair_h_ij_window"] = _window_series(pairwise_cbf_history[:, pair_index], event_step)
    event["pair_clearance_window"] = _window_series(pairwise_clearance_history[:, pair_index], event_step)
    return event


def _deadlock_summary(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    goal_tolerance: float,
    velocity_tolerance: float,
    dt: float,
) -> dict:
    window_steps = max(10, int(np.ceil(1.0 / max(dt, 1e-9))))
    speed_threshold = max(velocity_tolerance, 0.05)
    displacement_threshold = max(goal_tolerance * 0.2, 0.03)
    deadlock_start_steps: dict[int, list[int]] = {}
    deadlock_agents_at_end: list[int] = []
    first_deadlock_step = None

    for agent in range(positions.shape[1]):
        starts: list[int] = []
        active = False
        for step in range(window_steps - 1, positions.shape[0]):
            start_step = step - window_steps + 1
            not_at_goal = float(np.linalg.norm(positions[step, agent] - goals[agent])) > goal_tolerance
            speed = float(np.linalg.norm(velocities[step, agent]))
            displacement = float(np.linalg.norm(positions[step, agent] - positions[start_step, agent]))
            deadlocked = not_at_goal and speed <= speed_threshold and displacement <= displacement_threshold
            if deadlocked and not active:
                starts.append(start_step)
                active = True
                if first_deadlock_step is None:
                    first_deadlock_step = start_step
            elif not deadlocked:
                active = False
        if active:
            deadlock_agents_at_end.append(agent)
        if starts:
            deadlock_start_steps[agent] = starts

    return {
        "deadlock_count": len(deadlock_agents_at_end),
        "deadlock_agents": deadlock_agents_at_end,
        "first_deadlock_step": first_deadlock_step,
        "deadlock_start_steps": deadlock_start_steps,
        "deadlock_window_steps": window_steps,
        "deadlock_speed_threshold": speed_threshold,
        "deadlock_displacement_threshold": displacement_threshold,
    }


def _stopped_not_at_goal_summary(
    positions: np.ndarray,
    controls: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    safety_buffer: float,
    goal_tolerance: float,
) -> dict:
    n_steps = controls.shape[0]
    n_agents = positions.shape[1]
    control_thresholds = np.maximum(0.05, 0.1 * accel_limits)
    near_clearance_threshold = max(goal_tolerance, 0.15)
    stopped_steps = np.zeros(n_agents, dtype=int)
    near_steps = np.zeros(n_agents, dtype=int)
    isolated_steps = np.zeros(n_agents, dtype=int)

    for step in range(n_steps):
        agent_clearances = _agent_min_clearances(positions[step], radii, safety_buffer)
        for agent in range(n_agents):
            not_at_goal = float(np.linalg.norm(positions[step, agent] - goals[agent])) > goal_tolerance
            almost_stopped = float(np.linalg.norm(controls[step, agent])) <= control_thresholds[agent]
            if not (not_at_goal and almost_stopped):
                continue
            stopped_steps[agent] += 1
            if agent_clearances[agent] <= near_clearance_threshold:
                near_steps[agent] += 1
            else:
                isolated_steps[agent] += 1

    return {
        "stopped_not_at_goal_steps": stopped_steps.tolist(),
        "stopped_not_at_goal_near_robots_steps": near_steps.tolist(),
        "stopped_not_at_goal_isolated_steps": isolated_steps.tolist(),
        "stopped_not_at_goal_control_thresholds": control_thresholds.astype(float).tolist(),
        "stopped_not_at_goal_near_clearance_threshold": near_clearance_threshold,
    }


def simulate_scenario(config: ScenarioConfig, controller_name: str) -> ExperimentResult:
    controller = _controller_lookup(controller_name)
    n_agents = config.positions.shape[0]
    pair_indices = _pair_indices(n_agents)
    pair_index_map = {pair: idx for idx, pair in enumerate(pair_indices)}
    initial_pairwise_clearance, initial_pairwise_cbf = _compute_pairwise_metrics(
        config.positions,
        config.velocities,
        config.accel_limits,
        config.radii,
        config.safety_buffer,
    )

    positions_history: list[np.ndarray] = [config.positions.copy()]
    velocities_history: list[np.ndarray] = [config.velocities.copy()]
    controls_history: list[np.ndarray] = []
    nominal_controls_history: list[np.ndarray] = []
    qp_times_history: list[np.ndarray] = []
    clearance_history: list[float] = [float(initial_pairwise_clearance.min()) if initial_pairwise_clearance.size else 0.0]
    cbf_history: list[float] = [float(initial_pairwise_cbf.min()) if initial_pairwise_cbf.size else 0.0]
    pairwise_clearance_history: list[np.ndarray] = [initial_pairwise_clearance.copy()]
    pairwise_cbf_history: list[np.ndarray] = [initial_pairwise_cbf.copy()]
    estimation_history = None
    termination_reason = "step_limit"
    active_neighbor_history: list[np.ndarray] = []
    qp_fallback_history: list[np.ndarray] = []
    qp_infeasible_history: list[np.ndarray] = []
    collision_after_clamping = False
    first_infeasible_event = None
    collision_event = None
    collision_step = None

    alpha_estimates = None
    if controller_name in {
        "uncertain_heterogeneous_barrier",
        "decentralized_uncertain_heterogeneous_barrier",
        "centralized_uncertain_heterogeneous_barrier",
    }:
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
        if controller_name in {
            "decentralized_heterogeneous_barrier",
            "decentralized_uncertain_heterogeneous_barrier",
        }:
            kwargs["speed_limits"] = config.speed_limits
        if controller_name in {
            "uncertain_heterogeneous_barrier",
            "decentralized_uncertain_heterogeneous_barrier",
            "centralized_uncertain_heterogeneous_barrier",
        }:
            kwargs["alpha_estimates"] = alpha_estimates

        step_result = controller(**kwargs)
        applied_control = step_result.control
        if controller_name != "nominal":
            unclamped_positions, _ = _step_dynamics(
                positions_history[-1],
                velocities_history[-1],
                step_result.control,
                config.speed_limits,
                config.dt,
            )
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
            clamped_positions, _ = _step_dynamics(
                positions_history[-1],
                velocities_history[-1],
                applied_control,
                config.speed_limits,
                config.dt,
            )
            unclamped_collision = float(
                pairwise_clearance(unclamped_positions, config.radii, config.safety_buffer).min()
            ) < 0.0
            clamped_collision = float(
                pairwise_clearance(clamped_positions, config.radii, config.safety_buffer).min()
            ) < 0.0
            collision_after_clamping = collision_after_clamping or (clamped_collision and not unclamped_collision)
        controls_history.append(applied_control.copy())
        nominal_controls_history.append(step_result.nominal.copy())
        qp_times_history.append(step_result.qp_times_ms.copy())
        if step_result.active_neighbor_counts is not None:
            active_neighbor_history.append(step_result.active_neighbor_counts.copy())
        if step_result.qp_fallback_counts is not None:
            qp_fallback_history.append(step_result.qp_fallback_counts.copy())
        if step_result.qp_infeasible_counts is not None:
            qp_infeasible_history.append(step_result.qp_infeasible_counts.copy())
        if step_result.first_infeasible_event is not None and first_infeasible_event is None:
            first_infeasible_event = {"step": int(step), **step_result.first_infeasible_event}
        next_positions, next_velocities = _step_dynamics(
            positions_history[-1],
            velocities_history[-1],
            applied_control,
            config.speed_limits,
            config.dt,
        )
        positions_history.append(next_positions)
        velocities_history.append(next_velocities)
        next_pairwise_clearance, next_pairwise_cbf = _compute_pairwise_metrics(
            next_positions,
            next_velocities,
            config.accel_limits,
            config.radii,
            config.safety_buffer,
        )
        pairwise_clearance_history.append(next_pairwise_clearance.copy())
        pairwise_cbf_history.append(next_pairwise_cbf.copy())
        clearance_history.append(float(next_pairwise_clearance.min()) if next_pairwise_clearance.size else 0.0)
        cbf_history.append(float(next_pairwise_cbf.min()) if next_pairwise_cbf.size else 0.0)
        if clearance_history[-1] < 0.0:
            collision_pair_idx = int(np.argmin(next_pairwise_clearance))
            collision_pair = pair_indices[collision_pair_idx]
            collision_step = step + 1
            collision_event = {
                "step": collision_step,
                "pair": [int(collision_pair[0]), int(collision_pair[1])],
                "pair_key": _pair_key(collision_pair),
            }
            termination_reason = "collision"
            break
        if controller_name in {
            "uncertain_heterogeneous_barrier",
            "decentralized_uncertain_heterogeneous_barrier",
            "centralized_uncertain_heterogeneous_barrier",
        }:
            alpha_estimates = update_alpha_estimates(
                alpha_estimates,
                applied_control,
                config.accel_limits,
                config.dt,
                config.estimate_gain,
                global_accel_upper_bound=float(np.max(config.accel_limits)),
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
    pairwise_clearance_history_array = np.asarray(pairwise_clearance_history, dtype=float)
    pairwise_cbf_history_array = np.asarray(pairwise_cbf_history, dtype=float)
    if estimation_history is not None:
        estimation_history = np.asarray(estimation_history, dtype=float)
    last_step = positions.shape[0] - 1
    active_neighbor_history_array = (
        np.asarray(active_neighbor_history, dtype=float) if active_neighbor_history else None
    )
    qp_fallback_history_array = (
        np.asarray(qp_fallback_history, dtype=float) if qp_fallback_history else None
    )
    qp_infeasible_history_array = (
        np.asarray(qp_infeasible_history, dtype=float) if qp_infeasible_history else None
    )

    completion_step = _task_completion_time(
        positions,
        velocities,
        config.goals,
        config.goal_tolerance,
        config.velocity_tolerance,
    )
    all_goals_reached = completion_step is not None
    control_deviation = np.linalg.norm(controls - nominal_controls, axis=2)
    deadlock = _deadlock_summary(
        positions,
        velocities,
        config.goals,
        config.goal_tolerance,
        config.velocity_tolerance,
        config.dt,
    )
    stopped_not_at_goal = _stopped_not_at_goal_summary(
        positions,
        controls,
        config.goals,
        config.accel_limits,
        config.radii,
        config.safety_buffer,
        config.goal_tolerance,
    )
    if first_infeasible_event is not None:
        first_infeasible_pair = tuple(first_infeasible_event["pair"])
        first_infeasible_event["pair_key"] = _pair_key(first_infeasible_pair)
        first_infeasible_event = _attach_pair_event_diagnostics(
            first_infeasible_event,
            pair_index_map.get(first_infeasible_pair),
            pairwise_cbf_history_array,
            pairwise_clearance_history_array,
            int(first_infeasible_event["step"]),
        )
    if collision_event is not None:
        collision_pair = tuple(collision_event["pair"])
        collision_event = _attach_pair_event_diagnostics(
            collision_event,
            pair_index_map.get(collision_pair),
            pairwise_cbf_history_array,
            pairwise_clearance_history_array,
            max(0, int(collision_event["step"]) - 1),
        )
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
        "deadlock_count": int(deadlock["deadlock_count"]),
        "deadlock_agents": list(deadlock["deadlock_agents"]),
        "first_deadlock_step": deadlock["first_deadlock_step"],
    }
    if active_neighbor_history_array is not None:
        summary["mean_active_neighbors"] = float(active_neighbor_history_array.mean())
        summary["max_active_neighbors"] = int(active_neighbor_history_array.max())
    if qp_fallback_history_array is not None:
        summary["qp_fallback_count"] = int(np.round(qp_fallback_history_array.sum()))
    if qp_infeasible_history_array is not None:
        summary["qp_infeasible_count"] = int(np.round(qp_infeasible_history_array.sum()))
    summary["collision_after_clamping"] = bool(collision_after_clamping)
    if first_infeasible_event is not None:
        summary["first_infeasible_step"] = int(first_infeasible_event["step"])
        summary["first_infeasible_pair"] = list(first_infeasible_event["pair"])
        summary["first_infeasible_h_ij"] = float(first_infeasible_event["pair_h_ij_at_event"])
        summary["first_infeasible_h_margin"] = float(first_infeasible_event["pair_h_margin_to_violation"])
        summary["infeasibility_before_collision"] = bool(
            collision_step is not None and int(first_infeasible_event["step"]) < int(collision_step)
        )
    else:
        summary["infeasibility_before_collision"] = False
    if collision_event is not None:
        summary["collision_step"] = int(collision_event["step"])
        summary["collision_pair"] = list(collision_event["pair"])
        summary["collision_pair_h_ij"] = float(collision_event["pair_h_ij_at_event"])
        summary["collision_pair_h_margin"] = float(collision_event["pair_h_margin_to_violation"])
    if estimation_history is not None:
        summary["final_mean_alpha_estimate_error"] = float(
            np.mean(np.abs(estimation_history[-1] - config.accel_limits.reshape(1, -1)))
        )
    pairwise_min_cbf = (
        pairwise_cbf_history_array.min(axis=0) if pairwise_cbf_history_array.size else np.empty(0, dtype=float)
    )
    pairwise_min_cbf_steps = (
        pairwise_cbf_history_array.argmin(axis=0) if pairwise_cbf_history_array.size else np.empty(0, dtype=int)
    )
    pairwise_min_clearance = (
        pairwise_clearance_history_array.min(axis=0)
        if pairwise_clearance_history_array.size
        else np.empty(0, dtype=float)
    )
    pairwise_min_clearance_steps = (
        pairwise_clearance_history_array.argmin(axis=0)
        if pairwise_clearance_history_array.size
        else np.empty(0, dtype=int)
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
            "pair_indices": [list(pair) for pair in pair_indices],
            "pairwise_min_cbf": {
                _pair_key(pair): float(pairwise_min_cbf[idx]) for idx, pair in enumerate(pair_indices)
            },
            "pairwise_min_cbf_step": {
                _pair_key(pair): int(pairwise_min_cbf_steps[idx]) for idx, pair in enumerate(pair_indices)
            },
            "pairwise_min_clearance": {
                _pair_key(pair): float(pairwise_min_clearance[idx]) for idx, pair in enumerate(pair_indices)
            },
            "pairwise_min_clearance_step": {
                _pair_key(pair): int(pairwise_min_clearance_steps[idx]) for idx, pair in enumerate(pair_indices)
            },
            "active_neighbor_counts": (
                active_neighbor_history_array.tolist() if active_neighbor_history_array is not None else None
            ),
            "qp_fallback_history": (
                qp_fallback_history_array.tolist() if qp_fallback_history_array is not None else None
            ),
            "qp_infeasible_history": (
                qp_infeasible_history_array.tolist() if qp_infeasible_history_array is not None else None
            ),
            "first_infeasible_event": first_infeasible_event,
            "collision_event": collision_event,
            "deadlock": deadlock,
            "stopped_not_at_goal": stopped_not_at_goal,
        },
    )

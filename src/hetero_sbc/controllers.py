from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from cvxopt import matrix, solvers

from .barriers import full_barrier_rhs, pairwise_safe_distance, strategy_c_constraint


solvers.options["show_progress"] = False


@dataclass
class ControlStep:
    control: np.ndarray
    nominal: np.ndarray
    cbf_values: np.ndarray
    qp_times_ms: np.ndarray
    estimates: np.ndarray | None = None
    active_neighbor_counts: np.ndarray | None = None
    qp_fallback_counts: np.ndarray | None = None
    qp_infeasible_counts: np.ndarray | None = None
    first_infeasible_event: dict | None = None


@dataclass
class LocalQPResult:
    control: np.ndarray
    elapsed_ms: float
    used_fallback: bool
    infeasible: bool


def nominal_goal_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    kp: float,
    kd: float,
) -> np.ndarray:
    return -kp * (positions - goals) - kd * velocities


def clip_rows(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    for i, limit in enumerate(limits):
        clipped[i] = np.clip(clipped[i], -limit, limit)
    return clipped


def _solve_local_qp(
    u_nom: np.ndarray,
    accel_limit: float,
    a_rows: list[np.ndarray],
    b_rows: list[float],
) -> LocalQPResult:
    start = time.perf_counter()
    g_rows = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ]
    h_rows = [accel_limit, accel_limit, accel_limit, accel_limit]
    g_rows.extend(a_rows)
    h_rows.extend(b_rows)

    g = np.asarray(g_rows, dtype=float)
    h = np.asarray(h_rows, dtype=float)
    nominal = np.clip(u_nom, -accel_limit, accel_limit)

    def feasible(point: np.ndarray) -> bool:
        return bool(np.all(g @ point <= h + 1e-9))

    if feasible(nominal):
        return LocalQPResult(
            control=nominal,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            used_fallback=False,
            infeasible=False,
        )

    p = matrix(2.0 * np.eye(2))
    q = matrix(-2.0 * nominal.reshape(-1, 1))
    try:
        solution = solvers.qp(p, q, matrix(g), matrix(h))
    except ValueError:
        solution = None
    if solution is not None and solution.get("status") == "optimal":
        control = np.asarray(solution["x"], dtype=float).reshape(2)
        if feasible(control):
            return LocalQPResult(
                control=control,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                used_fallback=False,
                infeasible=False,
            )

    candidates: list[np.ndarray] = []
    for row, bound in zip(g, h):
        denom = float(row.dot(row))
        if denom <= 1e-12:
            continue
        projected = nominal - row * ((float(row.dot(nominal)) - bound) / denom)
        if feasible(projected):
            candidates.append(projected)

    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            mat = np.vstack([g[i], g[j]])
            if abs(np.linalg.det(mat)) <= 1e-12:
                continue
            point = np.linalg.solve(mat, np.array([h[i], h[j]], dtype=float))
            if feasible(point):
                candidates.append(point)

    if not candidates:
        control = nominal
        infeasible = True
    else:
        control = min(candidates, key=lambda point: float(np.linalg.norm(point - nominal)))
        infeasible = False
    return LocalQPResult(
        control=control,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        used_fallback=True,
        infeasible=infeasible,
    )


def _neighbor_radius(
    alpha_i: float,
    alpha_min: float,
    alpha_max: float,
    beta_i: float,
    beta_max: float,
    gamma_i: float,
    safety_buffer: float,
) -> float:
    # The paper's reduced neighborhood disk depends on heterogeneous
    # acceleration and speed bounds plus each agent's gamma value.
    braking_horizon = (2.0 * (alpha_i + alpha_max) / max(gamma_i, 1e-6)) ** (1.0 / 3.0)
    envelope = braking_horizon + beta_i + beta_max
    return safety_buffer + envelope * envelope / (2.0 * max(alpha_i + alpha_min, 1e-6))


def _barrier_filtered_control(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
    neighbor_alpha_estimates: np.ndarray | None = None,
    symmetric_weighting: bool = False,
    speed_limits: np.ndarray | None = None,
    use_neighborhood_reduction: bool = False,
) -> ControlStep:
    n_agents = positions.shape[0]
    nominal = nominal_goal_controller(positions, velocities, goals, kp, kd)
    nominal = clip_rows(nominal, accel_limits)

    controls = np.zeros_like(nominal)
    qp_times = np.zeros(n_agents, dtype=float)
    active_neighbor_counts = np.zeros(n_agents, dtype=float)
    qp_fallback_counts = np.zeros(n_agents, dtype=float)
    qp_infeasible_counts = np.zeros(n_agents, dtype=float)
    cbf_values = []
    first_infeasible_event: dict | None = None
    alpha_min = float(np.min(accel_limits))
    alpha_max = float(np.max(accel_limits))
    local_speed_limits = (
        np.asarray(speed_limits, dtype=float)
        if speed_limits is not None
        else np.full(n_agents, np.max(np.linalg.norm(velocities, axis=1)), dtype=float)
    )
    beta_max = float(np.max(local_speed_limits)) if local_speed_limits.size else 0.0
    for i in range(n_agents):
        a_rows: list[np.ndarray] = []
        b_rows: list[float] = []
        neighbor_ids: list[int] = []
        local_h_values: list[float] = []
        neighbor_radius = None
        if use_neighborhood_reduction:
            neighbor_radius = _neighbor_radius(
                accel_limits[i],
                alpha_min,
                alpha_max,
                local_speed_limits[i],
                beta_max,
                gamma[i],
                safety_buffer,
            )
        for j in range(n_agents):
            if i == j:
                continue
            if neighbor_radius is not None and float(np.linalg.norm(positions[i] - positions[j])) > neighbor_radius:
                continue
            alpha_j = accel_limits[j] if neighbor_alpha_estimates is None else neighbor_alpha_estimates[i, j]
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            weight = 0.5 if symmetric_weighting else None
            a_row, b_row, h_ij = strategy_c_constraint(
                positions[i],
                velocities[i],
                positions[j],
                velocities[j],
                accel_limits[i],
                alpha_j,
                gamma[i],
                safe_distance,
                weight=weight,
            )
            a_rows.append(a_row)
            b_rows.append(b_row)
            neighbor_ids.append(j)
            local_h_values.append(float(h_ij))
            cbf_values.append(h_ij)
        active_neighbor_counts[i] = len(a_rows)
        local_result = _solve_local_qp(nominal[i], accel_limits[i], a_rows, b_rows)
        controls[i] = local_result.control
        qp_times[i] = local_result.elapsed_ms
        qp_fallback_counts[i] = float(local_result.used_fallback)
        qp_infeasible_counts[i] = float(local_result.infeasible)
        if local_result.infeasible and first_infeasible_event is None and a_rows:
            nominal_slacks = np.asarray(
                [b_row - float(a_row.dot(nominal[i])) for a_row, b_row in zip(a_rows, b_rows)],
                dtype=float,
            )
            critical_idx = int(np.argmin(nominal_slacks))
            first_infeasible_event = {
                "agent": int(i),
                "neighbor": int(neighbor_ids[critical_idx]),
                "pair": [int(i), int(neighbor_ids[critical_idx])],
                "active_neighbors": int(len(a_rows)),
                "agent_position": positions[i].astype(float).tolist(),
                "neighbor_position": positions[neighbor_ids[critical_idx]].astype(float).tolist(),
                "agent_velocity": velocities[i].astype(float).tolist(),
                "neighbor_velocity": velocities[neighbor_ids[critical_idx]].astype(float).tolist(),
                "agent_goal": goals[i].astype(float).tolist(),
                "neighbor_goal": goals[neighbor_ids[critical_idx]].astype(float).tolist(),
                "agent_nominal_control": nominal[i].astype(float).tolist(),
                "nominal_slack": float(nominal_slacks[critical_idx]),
                "constraint_rhs": float(b_rows[critical_idx]),
                "constraint_at_nominal": float(a_rows[critical_idx].dot(nominal[i])),
                "pair_h_ij": float(local_h_values[critical_idx]),
                "agent_gamma": float(gamma[i]),
                "agent_accel_limit": float(accel_limits[i]),
                "neighbor_accel_estimate": float(
                    accel_limits[neighbor_ids[critical_idx]]
                    if neighbor_alpha_estimates is None
                    else neighbor_alpha_estimates[i, neighbor_ids[critical_idx]]
                ),
                "neighbor_distance": float(np.linalg.norm(positions[i] - positions[neighbor_ids[critical_idx]])),
                "neighbor_radius": (float(neighbor_radius) if neighbor_radius is not None else None),
                "min_local_cbf": float(min(local_h_values)),
            }

    return ControlStep(
        control=controls,
        nominal=nominal,
        cbf_values=np.asarray(cbf_values, dtype=float),
        qp_times_ms=qp_times,
        estimates=neighbor_alpha_estimates.copy() if neighbor_alpha_estimates is not None else None,
        active_neighbor_counts=active_neighbor_counts,
        qp_fallback_counts=qp_fallback_counts,
        qp_infeasible_counts=qp_infeasible_counts,
        first_infeasible_event=first_infeasible_event,
    )


def nominal_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    kp: float,
    kd: float,
) -> ControlStep:
    nominal = nominal_goal_controller(positions, velocities, goals, kp, kd)
    control = clip_rows(nominal, accel_limits)
    return ControlStep(
        control=control,
        nominal=control.copy(),
        cbf_values=np.empty(0, dtype=float),
        qp_times_ms=np.zeros(positions.shape[0], dtype=float),
    )


def decentralized_heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
    speed_limits: np.ndarray | None = None,
) -> ControlStep:
    return _barrier_filtered_control(
        positions,
        velocities,
        goals,
        accel_limits,
        radii,
        gamma,
        safety_buffer,
        kp,
        kd,
        speed_limits=speed_limits,
        use_neighborhood_reduction=True,
    )


def heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
) -> ControlStep:
    return centralized_heterogeneous_barrier_controller(
        positions,
        velocities,
        goals,
        accel_limits,
        radii,
        gamma,
        safety_buffer,
        kp,
        kd,
    )


def centralized_heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
) -> ControlStep:
    start = time.perf_counter()
    n_agents = positions.shape[0]
    nominal = nominal_goal_controller(positions, velocities, goals, kp, kd)
    nominal = clip_rows(nominal, accel_limits)

    g_rows: list[np.ndarray] = []
    h_rows: list[float] = []
    cbf_values: list[float] = []
    for i in range(n_agents):
        row = np.zeros(2 * n_agents, dtype=float)
        row[2 * i] = 1.0
        g_rows.append(row)
        h_rows.append(accel_limits[i])
        row = np.zeros(2 * n_agents, dtype=float)
        row[2 * i] = -1.0
        g_rows.append(row)
        h_rows.append(accel_limits[i])
        row = np.zeros(2 * n_agents, dtype=float)
        row[2 * i + 1] = 1.0
        g_rows.append(row)
        h_rows.append(accel_limits[i])
        row = np.zeros(2 * n_agents, dtype=float)
        row[2 * i + 1] = -1.0
        g_rows.append(row)
        h_rows.append(accel_limits[i])

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            delta_p = positions[i] - positions[j]
            delta_v = velocities[i] - velocities[j]
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            cbf_values.append(
                strategy_c_constraint(
                    positions[i],
                    velocities[i],
                    positions[j],
                    velocities[j],
                    accel_limits[i],
                    accel_limits[j],
                    gamma[i],
                    safe_distance,
                )[2]
            )
            row = np.zeros(2 * n_agents, dtype=float)
            row[2 * i : 2 * i + 2] = -delta_p
            row[2 * j : 2 * j + 2] = delta_p
            g_rows.append(row)
            h_rows.append(
                full_barrier_rhs(
                    delta_p,
                    delta_v,
                    accel_limits[i],
                    accel_limits[j],
                    gamma[i],
                    safe_distance,
                )
            )

    p = matrix(2.0 * np.eye(2 * n_agents))
    q = matrix(-2.0 * nominal.reshape(-1, 1))
    g = matrix(np.asarray(g_rows, dtype=float))
    h = matrix(np.asarray(h_rows, dtype=float))
    solution = solvers.qp(p, q, g, h)
    if solution.get("status") == "optimal":
        control = np.asarray(solution["x"], dtype=float).reshape(n_agents, 2)
    else:
        control = nominal

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ControlStep(
        control=control,
        nominal=nominal,
        cbf_values=np.asarray(cbf_values, dtype=float),
        qp_times_ms=np.full(n_agents, elapsed_ms / n_agents, dtype=float),
    )


def symmetric_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
) -> ControlStep:
    return _barrier_filtered_control(
        positions,
        velocities,
        goals,
        accel_limits,
        radii,
        gamma,
        safety_buffer,
        kp,
        kd,
        symmetric_weighting=True,
    )


def decentralized_uncertain_heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
    alpha_estimates: np.ndarray,
    speed_limits: np.ndarray | None = None,
) -> ControlStep:
    return _barrier_filtered_control(
        positions,
        velocities,
        goals,
        accel_limits,
        radii,
        gamma,
        safety_buffer,
        kp,
        kd,
        neighbor_alpha_estimates=alpha_estimates,
        speed_limits=speed_limits,
        use_neighborhood_reduction=True,
    )


def uncertain_heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
    alpha_estimates: np.ndarray,
) -> ControlStep:
    return centralized_uncertain_heterogeneous_barrier_controller(
        positions,
        velocities,
        goals,
        accel_limits,
        radii,
        gamma,
        safety_buffer,
        kp,
        kd,
        alpha_estimates,
    )


def centralized_uncertain_heterogeneous_barrier_controller(
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    accel_limits: np.ndarray,
    radii: np.ndarray,
    gamma: np.ndarray,
    safety_buffer: float,
    kp: float,
    kd: float,
    alpha_estimates: np.ndarray,
) -> ControlStep:
    start = time.perf_counter()
    n_agents = positions.shape[0]
    nominal = nominal_goal_controller(positions, velocities, goals, kp, kd)
    nominal = clip_rows(nominal, accel_limits)

    g_rows: list[np.ndarray] = []
    h_rows: list[float] = []
    cbf_values: list[float] = []
    for i in range(n_agents):
        for sign, offset in ((1.0, 0), (-1.0, 0), (1.0, 1), (-1.0, 1)):
            row = np.zeros(2 * n_agents, dtype=float)
            row[2 * i + offset] = sign
            g_rows.append(row)
            h_rows.append(accel_limits[i])

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            estimated_alpha_j = min(alpha_estimates[i, j], alpha_estimates[j, i])
            delta_p = positions[i] - positions[j]
            delta_v = velocities[i] - velocities[j]
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            cbf_values.append(
                strategy_c_constraint(
                    positions[i],
                    velocities[i],
                    positions[j],
                    velocities[j],
                    accel_limits[i],
                    estimated_alpha_j,
                    gamma[i],
                    safe_distance,
                )[2]
            )
            row = np.zeros(2 * n_agents, dtype=float)
            row[2 * i : 2 * i + 2] = -delta_p
            row[2 * j : 2 * j + 2] = delta_p
            g_rows.append(row)
            h_rows.append(
                full_barrier_rhs(
                    delta_p,
                    delta_v,
                    accel_limits[i],
                    estimated_alpha_j,
                    gamma[i],
                    safe_distance,
                )
            )

    p = matrix(2.0 * np.eye(2 * n_agents))
    q = matrix(-2.0 * nominal.reshape(-1, 1))
    g = matrix(np.asarray(g_rows, dtype=float))
    h = matrix(np.asarray(h_rows, dtype=float))
    solution = solvers.qp(p, q, g, h)
    if solution.get("status") == "optimal":
        control = np.asarray(solution["x"], dtype=float).reshape(n_agents, 2)
    else:
        control = nominal

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ControlStep(
        control=control,
        nominal=nominal,
        cbf_values=np.asarray(cbf_values, dtype=float),
        qp_times_ms=np.full(n_agents, elapsed_ms / n_agents, dtype=float),
        estimates=alpha_estimates.copy(),
    )


def update_alpha_estimates(
    current_estimates: np.ndarray,
    observed_controls: np.ndarray,
    self_accel_limits: np.ndarray,
    dt: float,
    gain: float,
    global_accel_upper_bound: float | None = None,
) -> np.ndarray:
    updated = current_estimates.copy()
    observed_norms = np.linalg.norm(observed_controls, axis=1)
    for i in range(current_estimates.shape[0]):
        for j in range(current_estimates.shape[1]):
            if i == j:
                updated[i, j] = self_accel_limits[j]
                continue
            target = max(updated[i, j], observed_norms[j])
            updated[i, j] = updated[i, j] + gain * dt * (target - updated[i, j])
            if global_accel_upper_bound is not None:
                updated[i, j] = min(updated[i, j], global_accel_upper_bound)
    return updated

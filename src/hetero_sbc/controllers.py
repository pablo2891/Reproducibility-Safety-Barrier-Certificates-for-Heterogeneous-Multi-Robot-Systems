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
) -> tuple[np.ndarray, float]:
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
        return nominal, (time.perf_counter() - start) * 1000.0

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
    else:
        control = min(candidates, key=lambda point: float(np.linalg.norm(point - nominal)))
    return control, (time.perf_counter() - start) * 1000.0


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
) -> ControlStep:
    n_agents = positions.shape[0]
    nominal = nominal_goal_controller(positions, velocities, goals, kp, kd)
    nominal = clip_rows(nominal, accel_limits)

    controls = np.zeros_like(nominal)
    qp_times = np.zeros(n_agents, dtype=float)
    cbf_values = []
    for i in range(n_agents):
        a_rows: list[np.ndarray] = []
        b_rows: list[float] = []
        for j in range(n_agents):
            if i == j:
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
            cbf_values.append(h_ij)
        controls[i], qp_times[i] = _solve_local_qp(nominal[i], accel_limits[i], a_rows, b_rows)

    return ControlStep(
        control=controls,
        nominal=nominal,
        cbf_values=np.asarray(cbf_values, dtype=float),
        qp_times_ms=qp_times,
        estimates=neighbor_alpha_estimates.copy() if neighbor_alpha_estimates is not None else None,
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
    true_accel_limits: np.ndarray,
    dt: float,
    gain: float,
) -> np.ndarray:
    updated = current_estimates.copy()
    observed_norms = np.linalg.norm(observed_controls, axis=1)
    for i in range(current_estimates.shape[0]):
        for j in range(current_estimates.shape[1]):
            if i == j:
                updated[i, j] = true_accel_limits[j]
                continue
            target = max(updated[i, j], observed_norms[j])
            updated[i, j] = updated[i, j] + gain * dt * (target - updated[i, j])
            updated[i, j] = min(updated[i, j], true_accel_limits[j])
    return updated

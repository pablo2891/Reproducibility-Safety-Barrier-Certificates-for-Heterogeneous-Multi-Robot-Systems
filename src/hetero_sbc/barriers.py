from __future__ import annotations

import math

import numpy as np


EPS = 1e-6


def pairwise_safe_distance(radii: np.ndarray, i: int, j: int, safety_buffer: float) -> float:
    return float(radii[i] + radii[j] + safety_buffer)


def pairwise_clearance(positions: np.ndarray, radii: np.ndarray, safety_buffer: float) -> np.ndarray:
    n_agents = positions.shape[0]
    clearances = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            safe_distance = pairwise_safe_distance(radii, i, j, safety_buffer)
            clearances.append(float(np.linalg.norm(positions[i] - positions[j]) - safe_distance))
    return np.asarray(clearances, dtype=float)


def cbf_value(
    delta_p: np.ndarray,
    delta_v: np.ndarray,
    alpha_i: float,
    alpha_j: float,
    safe_distance: float,
) -> float:
    distance = max(float(np.linalg.norm(delta_p)), EPS)
    gap = max(distance - safe_distance, EPS)
    braking_term = math.sqrt(2.0 * max(alpha_i + alpha_j, EPS) * gap)
    closing_speed = float(delta_p.dot(delta_v) / distance)
    return braking_term + closing_speed


def full_barrier_rhs(
    delta_p: np.ndarray,
    delta_v: np.ndarray,
    alpha_i: float,
    alpha_j: float,
    gamma: float,
    safe_distance: float,
) -> float:
    distance = max(float(np.linalg.norm(delta_p)), EPS)
    gap = max(distance - safe_distance, EPS)
    h_ij = cbf_value(delta_p, delta_v, alpha_i, alpha_j, safe_distance)
    sqrt_term = math.sqrt(2.0 * max(alpha_i + alpha_j, EPS) * gap)
    projection = float(delta_v.dot(delta_p))
    return (
        gamma * h_ij**3 / distance
        - (projection**2) / (distance**2)
        + float(delta_v.dot(delta_v))
        + (alpha_i + alpha_j) * projection / max(sqrt_term, EPS)
    )


def strategy_c_constraint(
    p_i: np.ndarray,
    v_i: np.ndarray,
    p_j: np.ndarray,
    v_j: np.ndarray,
    alpha_i: float,
    alpha_j: float,
    gamma_i: float,
    safe_distance: float,
    weight: float | None = None,
) -> tuple[np.ndarray, float, float]:
    delta_p = p_i - p_j
    delta_v = v_i - v_j
    distance = max(float(np.linalg.norm(delta_p)), EPS)
    gap = max(distance - safe_distance, EPS)
    total_alpha = max(alpha_i + alpha_j, EPS)
    if weight is None:
        weight = alpha_i / total_alpha

    h_ij = cbf_value(delta_p, delta_v, alpha_i, alpha_j, safe_distance)
    projection = float(delta_p.dot(delta_v))
    local_terms = projection / (distance**2) * float(delta_p.dot(v_i)) - float(delta_v.dot(v_i))
    rhs = weight * (
        gamma_i * h_ij**3 / distance
        + math.sqrt(total_alpha) * projection / math.sqrt(2.0 * gap)
    ) - local_terms
    return -delta_p.astype(float), float(rhs), float(h_ij)


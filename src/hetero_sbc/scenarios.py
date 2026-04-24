from __future__ import annotations

import math

import numpy as np

from .config import ScenarioConfig


def circle_swap(
    n_agents: int,
    radius: float,
    agile_alpha: float,
    cumbersome_alpha: float,
    agile_radius: float,
    cumbersome_radius: float,
    speed_limit: float,
    gamma: float,
    large_agent_fraction: float = 0.2,
    safety_buffer: float = 0.0,
    steps: int = 600,
    dt: float = 0.05,
) -> ScenarioConfig:
    angles = np.linspace(0.0, 2.0 * math.pi, n_agents, endpoint=False)
    positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    goals = -positions
    velocities = np.zeros_like(positions)

    n_large = max(1, int(round(n_agents * large_agent_fraction)))
    large_indices = np.linspace(0, n_agents - 1, n_large, dtype=int)
    accel_limits = np.full(n_agents, agile_alpha, dtype=float)
    radii = np.full(n_agents, agile_radius, dtype=float)
    accel_limits[large_indices] = cumbersome_alpha
    radii[large_indices] = cumbersome_radius
    speed_limits = np.full(n_agents, speed_limit, dtype=float)
    gamma_values = np.full(n_agents, gamma, dtype=float)

    return ScenarioConfig(
        name=f"circle_swap_{n_agents}",
        positions=positions,
        velocities=velocities,
        goals=goals,
        accel_limits=accel_limits,
        speed_limits=speed_limits,
        radii=radii,
        gamma=gamma_values,
        dt=dt,
        steps=steps,
        kp=1.0,
        kd=1.8,
        safety_buffer=safety_buffer,
        estimate_floor=min(cumbersome_alpha, agile_alpha) * 0.5,
    )


def baseline_six() -> ScenarioConfig:
    return lane_swap(
        n_agents=6,
        lane_x=1.8,
        lane_height=0.9,
        agile_alpha=1.2,
        cumbersome_alpha=0.6,
        agile_radius=0.2,
        cumbersome_radius=0.4,
        speed_limit=0.6,
        gamma=1.0,
        steps=600,
        large_agent_fraction=1 / 6,
        run_until_complete=True,
    )


def demo_small() -> ScenarioConfig:
    return lane_swap(
        n_agents=4,
        lane_x=1.1,
        lane_height=0.6,
        agile_alpha=1.2,
        cumbersome_alpha=0.6,
        agile_radius=0.16,
        cumbersome_radius=0.28,
        speed_limit=0.6,
        gamma=1.0,
        steps=220,
        large_agent_fraction=0.25,
    )


def scalability_case(n_agents: int) -> ScenarioConfig:
    return lane_swap(
        n_agents=n_agents,
        lane_x=2.2,
        lane_height=1.0 + 0.05 * n_agents,
        agile_alpha=1.2,
        cumbersome_alpha=0.6,
        agile_radius=0.16,
        cumbersome_radius=0.28,
        speed_limit=0.7,
        gamma=1.0,
        large_agent_fraction=0.25,
        steps=600,
        run_until_complete=True,
    )


def sensitivity_cases(
    ds_values: list[float],
    gamma_values: list[float],
) -> list[ScenarioConfig]:
    cases: list[ScenarioConfig] = []
    for ds in ds_values:
        for gamma in gamma_values:
            cfg = baseline_six()
            cfg.name = f"sensitivity_ds_{ds:.2f}_gamma_{gamma:.2f}"
            cfg.gamma[:] = gamma
            cfg.safety_buffer = ds
            cases.append(cfg)
    return cases


def uncertainty_case() -> ScenarioConfig:
    cfg = baseline_six()
    cfg.name = "uncertainty_baseline_six"
    cfg.estimate_floor = 0.2
    cfg.estimate_gain = 2.5
    return cfg


def named_scenario(name: str) -> ScenarioConfig:
    normalized = name.strip().lower()
    if normalized in {"demo", "demo_small", "small"}:
        return demo_small()
    if normalized in {"baseline", "baseline_six", "lane_swap_6", "paper_baseline"}:
        return baseline_six()
    if normalized in {"uncertainty", "uncertainty_baseline_six"}:
        return uncertainty_case()
    if normalized.startswith("scalability_"):
        _, count = normalized.split("_", maxsplit=1)
        return scalability_case(int(count))
    raise ValueError(
        "Unknown scenario name. Use one of: demo, baseline, uncertainty, or scalability_<count>."
    )


def lane_swap(
    n_agents: int,
    lane_x: float,
    lane_height: float,
    agile_alpha: float,
    cumbersome_alpha: float,
    agile_radius: float,
    cumbersome_radius: float,
    speed_limit: float,
    gamma: float,
    large_agent_fraction: float = 0.2,
    safety_buffer: float = 0.0,
    steps: int = 500,
    dt: float = 0.05,
    reverse_goal_order: bool = True,
    run_until_complete: bool = False,
) -> ScenarioConfig:
    left_count = (n_agents + 1) // 2
    right_count = n_agents // 2
    left_y = np.linspace(-lane_height, lane_height, left_count)
    right_y = np.linspace(-lane_height, lane_height, right_count)
    positions = np.zeros((n_agents, 2), dtype=float)
    positions[:left_count, 0] = -lane_x
    positions[:left_count, 1] = left_y
    positions[left_count:, 0] = lane_x
    positions[left_count:, 1] = right_y

    goals = np.zeros_like(positions)
    goals[:left_count, 0] = lane_x
    goals[left_count:, 0] = -lane_x
    if reverse_goal_order:
        goals[:left_count, 1] = np.linspace(lane_height, -lane_height, left_count)
        goals[left_count:, 1] = np.linspace(lane_height, -lane_height, right_count)
    else:
        goals[:left_count, 1] = left_y
        goals[left_count:, 1] = right_y

    velocities = np.zeros_like(positions)
    n_large = max(1, int(round(n_agents * large_agent_fraction)))
    large_indices = np.linspace(0, n_agents - 1, n_large, dtype=int)
    accel_limits = np.full(n_agents, agile_alpha, dtype=float)
    radii = np.full(n_agents, agile_radius, dtype=float)
    accel_limits[large_indices] = cumbersome_alpha
    radii[large_indices] = cumbersome_radius
    speed_limits = np.full(n_agents, speed_limit, dtype=float)
    gamma_values = np.full(n_agents, gamma, dtype=float)

    return ScenarioConfig(
        name=f"lane_swap_{n_agents}",
        positions=positions,
        velocities=velocities,
        goals=goals,
        accel_limits=accel_limits,
        speed_limits=speed_limits,
        radii=radii,
        gamma=gamma_values,
        dt=dt,
        steps=steps,
        run_until_complete=run_until_complete,
        kp=1.0,
        kd=1.8,
        safety_buffer=safety_buffer,
        estimate_floor=min(cumbersome_alpha, agile_alpha) * 0.5,
    )

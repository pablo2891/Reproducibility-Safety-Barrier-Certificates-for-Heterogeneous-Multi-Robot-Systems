"""Simulation and experiment utilities for heterogeneous safety barrier certificates."""

from __future__ import annotations

from typing import Any


__all__ = ["animate_robotarium_style", "named_scenario", "run_experiment_suite", "simulate_scenario"]


def __getattr__(name: str) -> Any:
    if name == "animate_robotarium_style":
        from .plotting import animate_robotarium_style

        return animate_robotarium_style
    if name == "named_scenario":
        from .scenarios import named_scenario

        return named_scenario
    if name == "run_experiment_suite":
        from .experiments import run_experiment_suite

        return run_experiment_suite
    if name == "simulate_scenario":
        from .simulator import simulate_scenario

        return simulate_scenario
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

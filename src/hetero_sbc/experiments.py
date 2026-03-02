from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentResult
from .plotting import plot_scalability, plot_sensitivity_grid, plot_time_series, plot_trajectories
from .scenarios import baseline_six, scalability_case, sensitivity_cases, uncertainty_case
from .simulator import simulate_scenario


def _write_summary_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_result(result: ExperimentResult, output_dir: Path) -> None:
    summary_path = output_dir / f"{result.name}_{result.controller}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": result.name,
                "controller": result.controller,
                "summary": result.summary,
                "metadata": result.metadata,
            },
            handle,
            indent=2,
        )
    plot_trajectories(result, output_dir / f"{result.name}_{result.controller}_trajectory.png")
    plot_time_series(result, output_dir / f"{result.name}_{result.controller}_timeseries.png")


def run_experiment_suite(output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    artifacts: dict[str, list[dict]] = {
        "baseline_comparison": [],
        "scalability": [],
        "sensitivity": [],
        "uncertainty": [],
    }

    baseline_cfg = baseline_six()
    for controller_name in ("nominal", "symmetric_barrier", "heterogeneous_barrier"):
        result = simulate_scenario(baseline_cfg, controller_name)
        _serialize_result(result, output_path)
        row = {"experiment": baseline_cfg.name, "controller": controller_name, **result.summary}
        rows.append(row)
        artifacts["baseline_comparison"].append(row)

    scalability_results: list[ExperimentResult] = []
    for n_agents in (10, 15, 20):
        cfg = scalability_case(n_agents)
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        _serialize_result(result, output_path)
        row = {"experiment": cfg.name, "controller": "heterogeneous_barrier", "n_agents": n_agents, **result.summary}
        rows.append(row)
        artifacts["scalability"].append(row)
        scalability_results.append(result)
    plot_scalability(scalability_results, output_path / "scalability.png")

    ds_values = [0.0, 0.05, 0.10]
    gamma_values = [0.5, 1.0, 2.0]
    sensitivity_grid_clearance = np.zeros((len(ds_values), len(gamma_values)), dtype=float)
    sensitivity_grid_qp = np.zeros_like(sensitivity_grid_clearance)
    for cfg in sensitivity_cases(ds_values, gamma_values):
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        _serialize_result(result, output_path)
        ds_idx = ds_values.index(cfg.safety_buffer)
        gamma_idx = gamma_values.index(float(cfg.gamma[0]))
        sensitivity_grid_clearance[ds_idx, gamma_idx] = result.summary["min_clearance"]
        sensitivity_grid_qp[ds_idx, gamma_idx] = result.summary["mean_qp_ms"]
        row = {"experiment": cfg.name, "controller": "heterogeneous_barrier", **result.summary}
        rows.append(row)
        artifacts["sensitivity"].append(row)
    plot_sensitivity_grid(ds_values, gamma_values, sensitivity_grid_clearance, "Minimum clearance [m]", output_path / "sensitivity_clearance.png")
    plot_sensitivity_grid(ds_values, gamma_values, sensitivity_grid_qp, "Mean QP solve time [ms]", output_path / "sensitivity_qp.png")

    uncertainty_cfg = uncertainty_case()
    uncertainty_result = simulate_scenario(uncertainty_cfg, "uncertain_heterogeneous_barrier")
    _serialize_result(uncertainty_result, output_path)
    uncertainty_row = {
        "experiment": uncertainty_cfg.name,
        "controller": "uncertain_heterogeneous_barrier",
        **uncertainty_result.summary,
    }
    rows.append(uncertainty_row)
    artifacts["uncertainty"].append(uncertainty_row)

    _write_summary_csv(rows, output_path / "experiment_summary.csv")
    with (output_path / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)
    return artifacts

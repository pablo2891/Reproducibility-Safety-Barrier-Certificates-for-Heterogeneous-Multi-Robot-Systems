from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentResult
from .plotting import plot_scalability, plot_sensitivity_grid, plot_time_series, plot_trajectories
from .scenarios import baseline_six, scalability_case, sensitivity_cases, uncertainty_case
from .simulator import simulate_scenario


SCALABILITY_AGENT_COUNTS = (10, 15, 20)


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


def _experiment_dir(output_dir: Path, category: str, experiment_name: str) -> Path:
    directory = output_dir / category / experiment_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _serialize_result(result: ExperimentResult, output_dir: Path, category: str) -> None:
    experiment_dir = _experiment_dir(output_dir, category, result.name)
    summary_path = experiment_dir / f"{result.controller}_summary.json"
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
    plot_trajectories(result, experiment_dir / f"{result.controller}_trajectory.png")
    plot_time_series(result, experiment_dir / f"{result.controller}_timeseries.png")


def run_experiment_suite(output_dir: str | Path) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_dir = output_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

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
        _serialize_result(result, output_path, "baseline")
        row = {"experiment": baseline_cfg.name, "controller": controller_name, **result.summary}
        rows.append(row)
        artifacts["baseline_comparison"].append(row)

    scalability_results: list[ExperimentResult] = []
    for n_agents in SCALABILITY_AGENT_COUNTS:
        cfg = scalability_case(n_agents)
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        _serialize_result(result, output_path, "scalability")
        row = {"experiment": cfg.name, "controller": "heterogeneous_barrier", "n_agents": n_agents, **result.summary}
        rows.append(row)
        artifacts["scalability"].append(row)
        scalability_results.append(result)
    (output_path / "scalability").mkdir(parents=True, exist_ok=True)
    plot_scalability(scalability_results, output_path / "scalability" / "scalability.png")

    ds_values = [0.0, 0.05, 0.10]
    gamma_values = [0.5, 1.0, 2.0]
    sensitivity_grid_clearance = np.zeros((len(ds_values), len(gamma_values)), dtype=float)
    sensitivity_grid_qp = np.zeros_like(sensitivity_grid_clearance)
    for cfg in sensitivity_cases(ds_values, gamma_values):
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        _serialize_result(result, output_path, "sensitivity")
        ds_idx = ds_values.index(cfg.safety_buffer)
        gamma_idx = gamma_values.index(float(cfg.gamma[0]))
        sensitivity_grid_clearance[ds_idx, gamma_idx] = result.summary["min_clearance"]
        sensitivity_grid_qp[ds_idx, gamma_idx] = result.summary["mean_qp_ms"]
        row = {"experiment": cfg.name, "controller": "heterogeneous_barrier", **result.summary}
        rows.append(row)
        artifacts["sensitivity"].append(row)
    (output_path / "sensitivity").mkdir(parents=True, exist_ok=True)
    plot_sensitivity_grid(
        ds_values,
        gamma_values,
        sensitivity_grid_clearance,
        "Minimum clearance [m]",
        output_path / "sensitivity" / "sensitivity_clearance.png",
    )
    plot_sensitivity_grid(
        ds_values,
        gamma_values,
        sensitivity_grid_qp,
        "Mean QP solve time [ms]",
        output_path / "sensitivity" / "sensitivity_qp.png",
    )

    uncertainty_cfg = uncertainty_case()
    uncertainty_result = simulate_scenario(uncertainty_cfg, "uncertain_heterogeneous_barrier")
    _serialize_result(uncertainty_result, output_path, "uncertainty")
    uncertainty_row = {
        "experiment": uncertainty_cfg.name,
        "controller": "uncertain_heterogeneous_barrier",
        **uncertainty_result.summary,
    }
    rows.append(uncertainty_row)
    artifacts["uncertainty"].append(uncertainty_row)

    _write_summary_csv(rows, summary_dir / "experiment_summary.csv")
    with (summary_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)
    return artifacts

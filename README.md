# Reproducibility-Safety-Barrier-Certificates-for-Heterogeneous-Multi-Robot-Systems

Reproduction scaffold for the paper "Safety Barrier Certificates for Heterogeneous Multi-Robot Systems" by Li Wang, Aaron D. Ames, and Magnus Egerstedt.

The codebase implements a deterministic simulation harness for the paper's double-integrator setup and reproduces the main qualitative and quantitative experiments:

- Baseline six-robot swap scenario.
- Scalability runs with `N = 10, 15, 20`.
- Sensitivity sweeps over safety distance buffer `D_s` and CBF parameter `gamma`.
- Unknown neighbor acceleration limits with conservative online estimates.
- Baselines: nominal controller without a safety filter and a naive symmetric barrier filter.

## Environment

The implementation uses Python 3.10+ and the following packages:

- `numpy`
- `scipy`
- `matplotlib`
- `cvxopt`

## Run

Set `PYTHONPATH=src` and run:

```bash
PYTHONPATH=src python3 scripts/run_experiments.py --output-dir results
```

To render a Robotarium-style animated simulation as a GIF:

```bash
PYTHONPATH=src python3 scripts/run_visual_simulation.py \
  --scenario baseline \
  --controller heterogeneous_barrier \
  --output results/baseline_visual.gif
```

This writes:

- `results/experiment_summary.csv`
- `results/experiment_summary.json`
- Per-run trajectory and time-series figures
- Aggregate scalability and sensitivity plots
- Optional Robotarium-style GIF animations

## Notes

- The simulation follows the paper's double-integrator model and uses a centralized heterogeneous barrier QP for the main reproduction runs, plus a naive symmetric barrier baseline and a conservative-estimate variant for unknown neighbor limits.
- The default geometry is a lane-swap scenario rather than an all-to-center circle, which keeps the dense multi-robot cases numerically tractable while preserving the swap-style conflict pattern.
- The scalability plot reports `p95` QP solve time to reduce sensitivity to solver jitter.
- Physical Robotarium deployment is not included yet; this repository currently covers the simulation side of the reproduction.
- Pairwise safety distance is modeled as `r_i + r_j + D_s`, so robot size heterogeneity is preserved while `D_s` is swept as an additional safety buffer.

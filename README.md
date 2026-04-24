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

This writes organized outputs under:

- `results/summary/`
- `results/baseline/`
- `results/scalability/`
- `results/sensitivity/`
- `results/uncertainty/`

To render a Robotarium-style animated simulation as a GIF:

```bash
PYTHONPATH=src python3 scripts/run_visual_simulation.py \
  --scenario demo \
  --controller heterogeneous_barrier
```

To render the larger scalability GIF set:

```bash
PYTHONPATH=src python3 scripts/run_visual_simulation.py \
  --scenario scalability_10 \
  --controller heterogeneous_barrier

PYTHONPATH=src python3 scripts/run_visual_simulation.py \
  --scenario scalability_15 \
  --controller heterogeneous_barrier

PYTHONPATH=src python3 scripts/run_visual_simulation.py \
  --scenario scalability_20 \
  --controller heterogeneous_barrier
```

This writes:

- `results/summary/experiment_summary.csv`
- `results/summary/experiment_summary.json`
- Per-run trajectory and time-series figures inside each experiment folder
- Aggregate plots inside `results/scalability/` and `results/sensitivity/`
- Optional GIF animations inside `results/visualizations/<scenario>/`

## Notes

- The simulation follows the paper's double-integrator model and uses a centralized heterogeneous barrier QP for the main reproduction runs, plus a naive symmetric barrier baseline and a conservative-estimate variant for unknown neighbor limits.
- The default geometry is a lane-swap scenario rather than an all-to-center circle, which keeps the dense multi-robot cases numerically tractable while preserving the swap-style conflict pattern.
- The default scalability sweep uses the larger original geometry with `N = 10, 15, 20` and now runs until each scalability case reaches the simulator's completion condition.
- The simulator applies a discrete-time safety clamp after the barrier controller so the continuous-time barrier policy remains robust under the finite integration step used for visualization and experiments.
- The scalability plot reports `p95` QP solve time to reduce sensitivity to solver jitter.
- Physical Robotarium deployment is not included yet; this repository currently covers the simulation side of the reproduction.
- Pairwise safety distance is modeled as `r_i + r_j + D_s`, so robot size heterogeneity is preserved while `D_s` is swept as an additional safety buffer.
- The `demo` scenario is a 4-robot lane swap with a longer horizon and shorter travel distance, intended to produce a small collision-free simulation that actually completes.

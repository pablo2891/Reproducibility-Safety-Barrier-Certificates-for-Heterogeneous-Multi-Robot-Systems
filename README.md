# Reproducibility-Safety-Barrier-Certificates-for-Heterogeneous-Multi-Robot-Systems

Reproduction scaffold for the paper "Safety Barrier Certificates for Heterogeneous Multi-Robot Systems" by Li Wang, Aaron D. Ames, and Magnus Egerstedt.

The codebase implements a deterministic simulation harness for the paper's double-integrator setup and reproduces the main qualitative and quantitative experiments:

- Canonical six-robot paper-style radial swap comparison between `nominal`, `decentralized_heterogeneous_barrier`, and `centralized_heterogeneous_barrier`.
- Paper-style radial scalability runs with `N = 6, 10, 15, 20`.
- Separate lane-swap scalability stress tests with `N = 10, 15, 20`.
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
- `results/paper_baseline/`
- `results/paper_scalability/`
- `results/stress_test/`
- `results/lane_swap_stress_tests/`
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

The canonical paper-style comparison is stored in `results/summary/experiment_summary.json`
under `paper_baseline_comparison`, `baseline_comparison`, and `canonical_swap_comparison`.
The radial paper-style scaling study is stored under `paper_scalability_comparison`.
The original six-robot lane-swap case is now stored separately under
`stress_test_lane_swap_comparison`.
The older lane-swap scalability study is stored separately under
`lane_swap_stress_test_scalability`.

## Notes

- The simulation follows the paper's double-integrator model and uses a centralized heterogeneous barrier QP for the main reproduction runs, plus a naive symmetric barrier baseline and a conservative-estimate variant for unknown neighbor limits.
- The repo now also exposes paper-style distributed/local-QP controller variants through the controller names `decentralized_heterogeneous_barrier` and `decentralized_uncertain_heterogeneous_barrier`. These solve one QP per agent rather than one joint QP for the whole swarm.
- The barrier formulas used by both the centralized and decentralized controllers now match the paper's `\gamma h^3 ||Δp||` scaling in the pairwise constraints.
- The decentralized controllers now use paper-style neighborhood reduction based on heterogeneous acceleration bounds, speed bounds, and per-agent `gamma`.
- A paper-style radial swap baseline is available as `paper_baseline_six`; it uses heterogeneous per-agent gamma values to break decentralized deadlocks in a way that matches the paper's qualitative use of `gamma` as a coordination priority.
- Paper-style radial scalability scenarios are available as `paper_scalability_<N>`. They extend `paper_baseline_six` by placing more robots on the same radial crossing layout while scaling the circle radius to preserve the six-robot baseline's adjacent chord spacing. For `N > 6`, they reuse the repo's smaller large-swarm robot radii so the radial study remains a meaningful scaling comparison instead of turning into an immediate packing artifact.
- The paper-faithful benchmark is now `paper_baseline_six`, a radial swap that better matches the paper's decentralized setup. It runs until one of three terminal conditions occurs: collision, all goals reached with low residual velocity, or the configured step budget is exhausted.
- The paper-faithful scaling study now uses those `paper_scalability_<N>` radial scenarios with `nominal`, `decentralized_heterogeneous_barrier`, and `centralized_heterogeneous_barrier` on the same family of setups.
- The default geometry is a lane-swap scenario rather than an all-to-center circle, which keeps the dense multi-robot cases numerically tractable while preserving the swap-style conflict pattern.
- Each run now logs `collision`, `all_goals_reached`, `completion_step`, `min_pair_distance`, and `termination_reason` in addition to the older timing and clearance metrics.
- Decentralized runs additionally log active-neighbor statistics, local-QP fallback and infeasibility counts, the first infeasible local-QP event and responsible pair when one exists, minimum pair distance, and whether a collision first appeared only after discrete-time clamping.
- The original `baseline_six` lane-swap case remains in the suite as a stress test rather than a paper baseline. In the current repo it is stricter than the paper-style radial swap for local-QP decentralization because multiple agents must resolve crossing traffic through a narrow lane geometry.
- The older lane-swap scalability sweep remains in the repo as an extra-harsh stress test for centralized behavior and dense crossing traffic. It is no longer the paper-faithful scaling study.
- The simulator applies a discrete-time safety clamp after the barrier controller so the continuous-time barrier policy remains robust under the finite integration step used for visualization and experiments.
- The canonical experiment suite still uses the validated `heterogeneous_barrier` controller path. The decentralized controller is available for inspection and iteration, but under the repo's current surrogate scenarios it is more conservative and may stall before completing the full swap.
- The scalability plot reports `p95` QP solve time to reduce sensitivity to solver jitter.
- Physical Robotarium deployment is not included yet; this repository currently covers the simulation side of the reproduction.
- Pairwise safety distance is modeled as `r_i + r_j + D_s`, so robot size heterogeneity is preserved while `D_s` is swept as an additional safety buffer.
- The `demo` scenario is a 4-robot lane swap with a longer horizon and shorter travel distance, intended to produce a small collision-free simulation that actually completes.

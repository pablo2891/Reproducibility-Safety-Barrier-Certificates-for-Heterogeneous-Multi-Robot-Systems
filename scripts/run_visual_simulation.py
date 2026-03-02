#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from hetero_sbc import animate_robotarium_style, named_scenario, simulate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Robotarium-style visual simulation as a GIF.")
    parser.add_argument(
        "--scenario",
        default="baseline",
        help="Scenario name: baseline, uncertainty, scalability_10, scalability_15, scalability_20",
    )
    parser.add_argument(
        "--controller",
        default="heterogeneous_barrier",
        choices=("nominal", "symmetric_barrier", "heterogeneous_barrier", "uncertain_heterogeneous_barrier"),
        help="Controller to simulate.",
    )
    parser.add_argument("--output", default="results/visual_simulation.gif", help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Render every k-th simulation frame.")
    parser.add_argument("--steps", type=int, default=None, help="Override scenario horizon.")
    args = parser.parse_args()

    config = named_scenario(args.scenario)
    if args.steps is not None:
        config.steps = args.steps
    if args.controller == "uncertain_heterogeneous_barrier" and args.scenario == "baseline":
        config = named_scenario("uncertainty")

    result = simulate_scenario(config, args.controller)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animate_robotarium_style(
        result=result,
        goals=config.goals,
        output_path=output_path,
        fps=args.fps,
        frame_skip=max(1, args.frame_skip),
    )
    print(output_path)


if __name__ == "__main__":
    main()

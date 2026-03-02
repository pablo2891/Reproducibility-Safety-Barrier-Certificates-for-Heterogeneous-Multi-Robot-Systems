#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hetero_sbc import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heterogeneous safety barrier certificate experiments.")
    parser.add_argument("--output-dir", default="results", help="Directory where figures and summaries are written.")
    args = parser.parse_args()

    artifacts = run_experiment_suite(Path(args.output_dir))
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()


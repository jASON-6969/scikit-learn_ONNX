"""Command-line entry points for individual benchmarks and tree-size studies."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig
from .experiment import run_experiment as run_experiment, verify_saved_run as verify_saved_run
from .reporting import generate_report
from .study import run_study, verify_study
from .study_reporting import generate_study_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Same-model CPU inference comparison")
    parser.add_argument("command", choices=("run", "report", "verify", "study", "study-report", "study-verify"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--trees", type=int, nargs="+", default=[10, 100, 500])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-calls", type=int, default=100)
    args = parser.parse_args(argv)
    output = args.output or Path("output/tree_study" if args.command.startswith("study") else "output/experiment")
    try:
        if args.command == "run":
            config = ExperimentConfig(warmup_calls=args.warmup, rounds=args.rounds, calls_per_round=args.calls)
            run_experiment(output, config)
        elif args.command == "report":
            generate_report(output)
        elif args.command == "study":
            config = ExperimentConfig(warmup_calls=args.warmup, rounds=args.rounds, calls_per_round=args.calls)
            run_study(output, config, tuple(args.trees), args.repeats, args.profile_calls)
        elif args.command == "study-report":
            generate_study_report(output)
        elif args.command == "study-verify":
            print(json.dumps(verify_study(output), indent=2))
        else:
            print(json.dumps(verify_saved_run(output), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


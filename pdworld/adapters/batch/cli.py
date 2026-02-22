from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdworld.core.constants import DEFAULT_SEEDS
from pdworld.adapters.batch.experiments import run_all_experiments, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PD-World Tabular TD experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all_parser = subparsers.add_parser("run-all", help="Run all experiments")
    run_all_parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Artifacts output root")
    run_all_parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="Seeds")

    run_exp_parser = subparsers.add_parser("run-exp", help="Run a single experiment")
    run_exp_parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True, help="Experiment id")
    run_exp_parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    run_exp_parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Artifacts output root")

    return parser


def cmd_run_all(args: argparse.Namespace) -> int:
    summary = run_all_experiments(output_root=args.output, seeds=args.seeds)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run_exp(args: argparse.Namespace) -> int:
    result, run_dir = run_experiment(exp_id=args.exp, seed=args.seed, output_root=args.output)
    payload = {
        "experiment": args.exp,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "terminal_hits": result.terminal_hits,
        "episodes_completed": len(result.episode_lengths),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-all":
        return cmd_run_all(args)
    if args.command == "run-exp":
        return cmd_run_exp(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

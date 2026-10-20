#!/usr/bin/env python3
"""Run a data-free, multi-seed SAF code-change audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MNSIM.Accuracy_Model.saf_statistics import (
    collect_saf_statistics,
    parse_seed_list,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument(
        "--seeds",
        required=True,
        nargs="+",
        help="One or more integer seeds; comma-separated values are accepted.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seeds = parse_seed_list(args.seeds)
        result = collect_saf_statistics(
            network=args.network,
            weights=args.weights,
            num_classes=args.num_classes,
            sim_config=args.sim_config,
            seeds=seeds,
        )
    except (OSError, TypeError, ValueError) as error:
        _parser().error(str(error))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote SAF statistics: {output_path}")
    for entry in result["per_seed"]:
        print(
            f"seed={entry['seed']} "
            f"changed={entry['changed_codes']}/{entry['total_codes']} "
            f"({entry['changed_fraction']:.6%})"
        )
    aggregate = result["aggregate"]
    print(
        "aggregate changed fraction: "
        f"mean={aggregate['changed_fraction_mean']:.6%} "
        f"std={aggregate['changed_fraction_std']:.6%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

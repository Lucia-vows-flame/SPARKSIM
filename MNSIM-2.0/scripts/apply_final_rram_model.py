#!/usr/bin/env python3
"""Apply one locked final RRAM model to an existing SimConfig."""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path

from MNSIM.Interface.final_rram_contract import (
    apply_final_rram_model,
    validate_final_rram_parser,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sim-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-model", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    base = Path(args.base_sim_config).resolve()
    output = Path(args.output).resolve()
    config = configparser.ConfigParser()
    if not config.read(base, encoding="UTF-8"):
        raise FileNotFoundError(f"cannot read base SimConfig: {base}")

    apply_final_rram_model(config, args.device_model, repo_root=repo_root)
    validate_final_rram_parser(config, source=str(output), repo_root=repo_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="UTF-8") as stream:
        config.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

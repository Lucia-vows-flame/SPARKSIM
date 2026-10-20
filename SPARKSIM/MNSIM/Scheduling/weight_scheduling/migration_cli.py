"""Command-line migration of trusted legacy schedule pickles to schema v2 JSON."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from MNSIM.Scheduling.weight_scheduling.legacy_migration import (
    migrate_legacy_schedule,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Trusted legacy .schedule.pt file; repeat for multiple files.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path (default: OUTPUT_DIR/migration_manifest.json).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _destination_for(source: Path, output_dir: Path) -> Path:
    experiment = source.parent.parent.name or "legacy"
    name = source.name[:-3] if source.name.endswith(".pt") else source.name
    return output_dir / f"{experiment}__{name}.schema-v2.json"


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "migration_id": "legacy_master_schedule_pt_to_schema_v2_json_v1",
                "records": records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    manifest_path = (args.manifest or output_dir / "migration_manifest.json").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    destinations: set[Path] = set()

    for raw_source in args.input:
        source = raw_source.resolve()
        destination = _destination_for(source, output_dir)
        if destination in destinations:
            _parser().error(f"multiple inputs map to the same destination: {destination}")
        destinations.add(destination)
        if destination.exists() and not args.overwrite:
            _parser().error(f"destination already exists: {destination}")

        try:
            artifact = migrate_legacy_schedule(
                source,
                destination,
                repo_root=args.repo_root,
            )
            record = {
                "status": "succeeded",
                "source_path": str(source),
                "source_sha256": artifact.migration_metadata["source_sha256"],
                "destination_path": str(destination),
                "destination_sha256": sha256_file(destination),
                "path_id": artifact.path_id,
                "policy_id": artifact.policy_id,
                "round_count": len(artifact.rounds),
                "execution_compatible": artifact.execution_compatible,
            }
            print(f"Migrated {source} -> {destination}", flush=True)
            del artifact
        except Exception as error:
            record = {
                "status": "failed",
                "source_path": str(source),
                "destination_path": str(destination),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"Failed {source}: {error}", flush=True)
            records.append(record)
            _write_manifest(manifest_path, records)
            gc.collect()
            if not args.continue_on_error:
                return 1
            continue

        records.append(record)
        _write_manifest(manifest_path, records)
        gc.collect()

    return 0 if all(item["status"] == "succeeded" for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

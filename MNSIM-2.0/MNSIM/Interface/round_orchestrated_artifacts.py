"""Durable JSON artifacts owned by the round-orchestrated runner."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from MNSIM.Interface.round_orchestrated_types import RunnerError


INVALID_OUTPUT_DIRECTORY = "INVALID_OUTPUT_DIRECTORY"

RUNNER_OWNED_ARTIFACTS = frozenset(
    {
        "run_manifest.json",
        "summary.json",
        "accuracy_report.json",
        "deployment_weights_9bit.pth",
        "deployment_weights_9bit_manifest.json",
        "readc_calibration.json",
        "round_plan.json",
        "phase2_report.json",
    }
)


def _fsync_directory(output_dir: Path) -> None:
    if os.name != "posix":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(output_dir, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class ArtifactStore:
    output_dir: Path

    @classmethod
    def open(cls, output_dir: Path, overwrite: bool) -> "ArtifactStore":
        output_dir = Path(output_dir).absolute()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RunnerError(
                INVALID_OUTPUT_DIRECTORY,
                f"cannot create output directory {output_dir}: {error}",
            ) from error

        if not output_dir.is_dir():
            raise RunnerError(
                INVALID_OUTPUT_DIRECTORY,
                f"output path is not a directory: {output_dir}",
            )

        try:
            is_nonempty = next(output_dir.iterdir(), None) is not None
        except OSError as error:
            raise RunnerError(
                INVALID_OUTPUT_DIRECTORY,
                f"cannot inspect output directory {output_dir}: {error}",
            ) from error

        if is_nonempty and not overwrite:
            raise RunnerError(
                INVALID_OUTPUT_DIRECTORY,
                f"output directory is not empty: {output_dir}",
            )
        if overwrite:
            cls._remove_owned_artifacts(output_dir)
        return cls(output_dir=output_dir)

    @staticmethod
    def _remove_owned_artifacts(output_dir: Path) -> None:
        removable_paths: list[Path] = []
        for artifact_name in sorted(RUNNER_OWNED_ARTIFACTS):
            artifact_path = output_dir / artifact_name
            try:
                mode = artifact_path.lstat().st_mode
            except FileNotFoundError:
                pass
            except OSError as error:
                raise RunnerError(
                    INVALID_OUTPUT_DIRECTORY,
                    f"cannot inspect runner artifact {artifact_path}: {error}",
                ) from error
            else:
                if not stat.S_ISREG(mode):
                    raise RunnerError(
                        INVALID_OUTPUT_DIRECTORY,
                        f"runner artifact is not a regular file: {artifact_path}",
                    )
                removable_paths.append(artifact_path)

            try:
                temporary_paths = list(
                    output_dir.glob(f".{artifact_name}.*.tmp")
                )
            except OSError as error:
                raise RunnerError(
                    INVALID_OUTPUT_DIRECTORY,
                    f"cannot inspect runner temporary artifacts in {output_dir}: {error}",
                ) from error
            for temporary_path in temporary_paths:
                try:
                    temporary_mode = temporary_path.lstat().st_mode
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise RunnerError(
                        INVALID_OUTPUT_DIRECTORY,
                        f"cannot inspect runner temporary artifact {temporary_path}: {error}",
                    ) from error
                if stat.S_ISREG(temporary_mode):
                    removable_paths.append(temporary_path)

        for artifact_path in removable_paths:
            try:
                artifact_path.unlink()
            except OSError as error:
                raise RunnerError(
                    INVALID_OUTPUT_DIRECTORY,
                    f"cannot remove runner artifact {artifact_path}: {error}",
                ) from error

    def write_json(self, artifact_name: str, payload: Any) -> Path:
        if artifact_name not in RUNNER_OWNED_ARTIFACTS:
            raise RunnerError(
                INVALID_OUTPUT_DIRECTORY,
                f"unknown runner artifact: {artifact_name}",
            )

        target_path = self.output_dir / artifact_name
        temporary_path: Path | None = None
        primary_error: BaseException | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_dir,
                prefix=f".{artifact_name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(
                    payload,
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target_path)
            _fsync_directory(self.output_dir)
            return target_path
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    if primary_error is None:
                        raise
                    add_note = getattr(primary_error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            f"temporary artifact cleanup failed: {cleanup_error}"
                        )


__all__ = [
    "INVALID_OUTPUT_DIRECTORY",
    "RUNNER_OWNED_ARTIFACTS",
    "ArtifactStore",
]

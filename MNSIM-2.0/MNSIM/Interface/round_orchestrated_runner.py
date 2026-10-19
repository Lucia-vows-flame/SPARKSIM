"""Public core for round-aware ANN simulation orchestration."""

from __future__ import annotations

import configparser
from copy import deepcopy
import io
import os
import pickle
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Accuracy_Model.readc import READC_PROVENANCE
from MNSIM.Accuracy_Model.readc_artifact import (
    ReADCArtifactIdentity,
    bit_weights_sha256,
    file_sha256,
)
from MNSIM.Accuracy_Model.prequantized_weights import (
    PrequantizedWeightError,
    prepare_prequantized_deployment,
)
from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    ADC_DIGITAL_LIF,
    FRONTEND_LUT_LIF_LUT,
    canonical_json_sha256,
)
from MNSIM.Accuracy_Model.snn_lif_perturbation import LIFPerturbationRuntime
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Interface.final_rram_contract import (
    FinalRramContractError,
    load_snn_accuracy_device_contract,
    validate_final_rram_simconfig,
)
from MNSIM.Interface.snn_hw_config import read_weight_quantization_bit
from MNSIM.Interface.round_orchestrated_artifacts import (
    INVALID_OUTPUT_DIRECTORY,
    ArtifactStore,
)
from MNSIM.Interface.round_orchestrated_metrics import (
    INVALID_MAC_ESTIMATE,
    count_model_operations,
    derive_hardware_rates,
)
from MNSIM.Interface.round_orchestrated_types import (
    ErrorRecord,
    RoundOrchestratedRunRequest,
    RoundOrchestratedRunResult,
    RunnerError,
    StageStatus,
)
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.phase1.config import build_hardware_projection
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.phase2.types import Phase2Invalid
from MNSIM.Scheduling.round_types import (
    Phase1Invalid,
    UNSUPPORTED_LAYER_TYPE,
    WeightedPhysicalRound,
    round_plan_from_dict,
    validate_round_plan,
)
from MNSIM.Scheduling.phase2.lowerer import RoundLowerer
from MNSIM.Scheduling.phase2.tcg_adapter import serialize_round_placement
from MNSIM.Scheduling.phase2.types import BoundaryInputView
from MNSIM.Scheduling.weight_scheduling import (
    FlashSeFPTRPolicy,
    LogicalTile,
    RoundSchedulingContext,
    SchedulingConstraints,
    SNNActivityProfileError,
    SNNFlashSeFPTRPolicy,
    SNNNativeTemporalPolicy,
    SequentialPolicy,
    build_problem_from_round_plan,
    canonical_sha256,
    generate_weight_schedule,
    generate_weight_schedule_from_custom,
    load_custom_schedule,
    load_device_model_from_sim_config,
    load_native_device_model_from_sim_config,
    load_schedule_artifact,
    save_schedule_artifact,
)
from MNSIM.Scheduling.round_types import SequenceSpec
from MNSIM.Scheduling.weight_scheduling.custom_schedule import remap_custom_schedule_placement
from round_orchestrated_interface import RoundOrchestratedTrainTestInterface


INVALID_STAGE_SELECTION = "INVALID_STAGE_SELECTION"
DATASET_MODULE_REQUIRED = "DATASET_MODULE_REQUIRED"
INVALID_NUM_CLASSES = "INVALID_NUM_CLASSES"
INVALID_ACCURACY_SEED = "INVALID_ACCURACY_SEED"
CHECKPOINT_LOAD_FAILED = "CHECKPOINT_LOAD_FAILED"
PHASE1_INVALID = "PHASE1_INVALID"
PHASE2_FAILED = "PHASE2_FAILED"
ACCURACY_FAILED = "ACCURACY_FAILED"
INVALID_FINAL_RRAM_MODEL = "INVALID_FINAL_RRAM_MODEL"

_DEFAULT_CHECKPOINT_ERROR_TYPES: tuple[type[BaseException], ...] = (
    pickle.UnpicklingError,
    EOFError,
    UnicodeDecodeError,
    zipfile.BadZipFile,
    tarfile.ReadError,
)

_TORCH_CHECKPOINT_RUNTIME_MARKERS = (
    "pytorchstreamreader failed reading zip archive",
    "pytorchstreamreader failed reading file",
    "pytorchstreamreader failed locating file",
    "pytorchstreamreader failed finding central directory",
    "invalid magic number; corrupt file?",
    "invalid protocol version",
    "failed to read record",
)

_ARTIFACT_KEYS = {
    "run_manifest.json": "run_manifest",
    "summary.json": "summary",
    "accuracy_report.json": "accuracy_report",
    "deployment_weights_9bit.pth": "deployment_weights_9bit",
    "deployment_weights_9bit_manifest.json": "deployment_weights_9bit_manifest",
    "readc_calibration.json": "readc_calibration",
    "round_plan.json": "round_plan",
    "phase2_report.json": "phase2_report",
    "schedule_artifact": "schedule_artifact",
}


def _build_phase1_projection(sim_config_path: str | Path) -> dict[str, Any]:
    """Build the Phase 1 resource view from the backend SimConfig."""

    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(sim_config_path, encoding="UTF-8")
    except (OSError, configparser.Error) as error:
        raise Phase1Invalid(
            "INVALID_SIM_CONFIG",
            f"could not read SimConfig path {sim_config_path}: {error}",
        ) from error
    if not loaded:
        raise Phase1Invalid(
            "INVALID_SIM_CONFIG",
            f"could not read SimConfig path {sim_config_path}",
        )

    try:
        projection = {
            "Xbar_Size": _sim_config_pair(parser, "Crossbar level", "Xbar_Size"),
            "Tile_Num": _sim_config_pair(parser, "Architecture level", "Tile_Num"),
            "PE_Num": _sim_config_pair(parser, "Tile level", "PE_Num"),
            "Group_Num": _sim_config_int(
                parser, "Process element level", "Group_Num"
            ),
            "Device_Level": _sim_config_int(
                parser,
                "Device level",
                "Device_Level",
            ),
            "Weight_precision": _sim_config_int(
                parser, "Algorithm Configuration", "Weight_precision"
            ),
            "Xbar_Polarity": _sim_config_int(
                parser, "Process element level", "Xbar_Polarity"
            ),
        }
    except (TypeError, ValueError, configparser.Error) as error:
        raise Phase1Invalid(
            "INVALID_SIM_CONFIG",
            f"invalid Phase 1 projection in SimConfig {sim_config_path}: {error}",
        ) from error

    build_hardware_projection(projection)
    return projection


def _sim_config_pair(
    parser: configparser.ConfigParser, section: str, option: str
) -> list[int]:
    value = parser.get(section, option)
    parts = [part.strip() for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise ValueError(f"{section}.{option} must contain two integers")
    return [int(part) for part in parts]


def _sim_config_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> int:
    return int(parser.get(section, option))


def _model_inputs_from_root_layers(model_ir: Any) -> dict[str, Any]:
    root_layers = [
        layer
        for layer in list(getattr(model_ir, "layers", []) or [])
        if not list(getattr(layer, "input_layer_ids", []) or [])
    ]
    if not root_layers:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            "ModelIR has no root layer from which to seed model_input",
        )

    shapes = [_root_model_input_shape(layer) for layer in root_layers]
    first_shape = shapes[0]
    if any(shape != first_shape for shape in shapes[1:]):
        raise Phase1Invalid(
            "INCOMPATIBLE_ROOT_MODEL_INPUT",
            "root ModelIR layers describe incompatible model_input shapes",
        )
    report = {
        "model_input": {
            "tensor_role": "activation",
            "shape": first_shape,
        }
    }
    return report


def _root_model_input_shape(layer: Any) -> dict[str, Any]:
    layer_id = str(getattr(layer, "layer_id", "<unknown>"))
    layer_type = str(getattr(layer, "layer_type", ""))
    shape = getattr(layer, "shape", None)
    if not isinstance(shape, Mapping):
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} shape must be a mapping",
        )

    if layer_type == "conv":
        return {
            "Inputchannel": _positive_root_input_int(
                shape, "Inputchannel", layer_id
            ),
            "Inputsize": _root_input_size(shape, layer_id),
            "Inputbit": _positive_root_input_int(shape, "Inputbit", layer_id),
        }
    if layer_type in {"fc", "fc_like_projection"}:
        return {
            "Infeature": _positive_root_input_int(shape, "Infeature", layer_id),
            "Inputbit": _positive_root_input_int(shape, "Inputbit", layer_id),
        }
    raise Phase1Invalid(
        "UNSUPPORTED_ROOT_MODEL_INPUT",
        f"root layer {layer_id} has unsupported input type {layer_type!r}",
    )


def _positive_root_input_int(
    shape: Mapping[str, Any], key: str, layer_id: str
) -> int:
    if key not in shape:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} is missing shape field {key}",
        )
    value = shape[key]
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} shape field {key} must be a positive integer",
        )
    return int(value)


def _root_input_size(shape: Mapping[str, Any], layer_id: str) -> list[int]:
    if "Inputsize" not in shape:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} is missing shape field Inputsize",
        )
    value = shape["Inputsize"]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} shape field Inputsize must contain two positive integers",
        )
    if len(value) != 2:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} shape field Inputsize must contain two positive integers",
        )
    return [
        _positive_root_input_value(value[0], "Inputsize[0]", layer_id),
        _positive_root_input_value(value[1], "Inputsize[1]", layer_id),
    ]


def _positive_root_input_value(value: Any, key: str, layer_id: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise Phase1Invalid(
            "INVALID_ROOT_MODEL_INPUT",
            f"root layer {layer_id} shape field {key} must be a positive integer",
        )
    return int(value)


@dataclass
class RunnerDependencies:
    """Test-only dependency injection for the public runner core."""

    interface_factory: Callable[..., Any] = RoundOrchestratedTrainTestInterface
    weight_update: Callable[..., Any] = weight_update
    model_ir_builder: Callable[..., Any] = build_model_ir_from_mnsim_network
    phase1_projection_builder: Callable[[str | Path], dict[str, Any]] = (
        _build_phase1_projection
    )
    plan_builder: Callable[..., Any] = build_round_plan
    phase2_executor: Callable[..., Any] = execute_round_plan
    final_rram_validator: Callable[[str | Path], dict[str, Any] | None] = (
        validate_final_rram_simconfig
    )
    static_backend: Any = None
    deployment_preparer: Callable[..., dict[str, Any]] = (
        prepare_prequantized_deployment
    )
    enable_prequantized_weight_flow: bool = False
    checkpoint_error_types: tuple[type[BaseException], ...] = (
        _DEFAULT_CHECKPOINT_ERROR_TYPES
    )
    checkpoint_error_predicate: Callable[[BaseException], bool] | None = None


def run_round_orchestrated_simulation(
    request: RoundOrchestratedRunRequest,
    *,
    dependencies: RunnerDependencies | None = None,
) -> RoundOrchestratedRunResult:
    """Run the requested accuracy and round-aware hardware stages without printing."""

    try:
        store = ArtifactStore.open(request.output_dir, overwrite=request.overwrite)
    except Exception as error:
        return _output_open_failure(request, error)

    requested_stages = request.requested_stages()
    result = RoundOrchestratedRunResult(
        status="running",
        exit_code=0,
        request=request,
        stage_statuses={
            stage_name: StageStatus(name=stage_name, status="pending")
            for stage_name in requested_stages
        },
    )
    _write_manifest(store, result)

    input_error = _validate_inputs(request, requested_stages)
    if input_error is not None:
        _record_top_level_error(result, input_error)
        _mark_requested_stages_not_run(result)
        return _finalize(store, result, exit_code=2)

    deps = dependencies or RunnerDependencies(enable_prequantized_weight_flow=True)
    try:
        final_rram_metadata = deps.final_rram_validator(request.sim_config)
        if (
            "hardware" in requested_stages
            and not request.schedule_only
            and final_rram_metadata is not None
            and not final_rram_metadata.get(
                "hardware_write_accounting_supported", True
            )
        ):
            raise FinalRramContractError(
                f"{final_rram_metadata.get('model_id', 'selected RRAM model')} "
                "does not provide a complete source-to-target write transition model; "
                "hardware write accounting is unavailable"
            )
    except FinalRramContractError as error:
        _record_top_level_error(
            result,
            ErrorRecord(
                code=INVALID_FINAL_RRAM_MODEL,
                detail=_exception_detail(error),
            ),
        )
        _mark_requested_stages_not_run(result)
        return _finalize(store, result, exit_code=2)

    is_snn = "snn" in request.network.lower()
    deployment_manifest = None
    deployment_checkpoint = store.output_dir / "deployment_weights_9bit.pth"
    deployment_manifest_path = (
        store.output_dir / "deployment_weights_9bit_manifest.json"
    )
    use_deployment_flow = bool(
        deps.enable_prequantized_weight_flow
        and is_snn
        and (request.run_pim or "hardware" in requested_stages)
    )
    try:
        if use_deployment_flow:
            weight_quantization_bit = read_weight_quantization_bit(str(request.sim_config))
            deployment_manifest = _quiet_call(
                deps.deployment_preparer,
                request.weights,
                deployment_checkpoint,
                deployment_manifest_path,
                qbit=weight_quantization_bit,
            )
            _register_existing_artifact(
                result, "deployment_weights_9bit.pth", deployment_checkpoint
            )
            _register_existing_artifact(
                result,
                "deployment_weights_9bit_manifest.json",
                deployment_manifest_path,
            )
            _write_manifest(store, result)

        context_kwargs = {
            "network_module": request.network,
            "dataset_module": _accuracy_dataset_module(request)
            if "accuracy" in requested_stages
            else None,
            "SimConfig_path": str(request.sim_config),
            "device": request.device,
            "num_classes": request.num_classes,
        }
        if use_deployment_flow:
            original_context = None
            if "accuracy" in requested_stages and request.run_original:
                original_context = _quiet_call(
                    deps.interface_factory,
                    **context_kwargs,
                    weights_file=str(request.weights),
                    use_prequantized_weights=False,
                )
            pim_context = _quiet_call(
                deps.interface_factory,
                **context_kwargs,
                weights_file=str(deployment_checkpoint),
                use_prequantized_weights=True,
                prequantized_manifest_file=str(deployment_manifest_path),
            )
            hardware_context = pim_context
        else:
            shared_context = _quiet_call(
                deps.interface_factory,
                **context_kwargs,
                weights_file=str(request.weights),
            )
            original_context = shared_context
            pim_context = shared_context
            hardware_context = shared_context
    except Exception as error:
        _record_top_level_error(
            result,
            ErrorRecord(
                code=_context_error_code(error, deps),
                detail=_exception_detail(error),
            ),
        )
        _mark_requested_stages_not_run(result)
        return _finalize(store, result, exit_code=2)

    fatal_hardware_preparation_failure = False
    if "accuracy" in requested_stages:
        _run_accuracy_stage(
            store,
            result,
            request,
            original_context,
            pim_context,
            deps,
            deployment_manifest=deployment_manifest,
        )
    if "hardware" in requested_stages:
        fatal_hardware_preparation_failure = _run_hardware_stage(
            store, result, request, hardware_context, deps
        )

    has_stage_failure = any(
        stage_status.status == "failed"
        for stage_status in result.stage_statuses.values()
    )
    if fatal_hardware_preparation_failure:
        exit_code = 2
    elif has_stage_failure:
        exit_code = 1
    else:
        exit_code = 0
    return _finalize(store, result, exit_code=exit_code)


def _accuracy_dataset_module(request: RoundOrchestratedRunRequest) -> str | None:
    """Select an orchestration adapter without changing the source dataset module."""

    module = request.dataset_module
    if module is None:
        return None
    if "snn" in request.network.lower() and module.endswith("dvs_binned_npz"):
        return "round_orchestrated_snn_dataset"
    if module == "MNSIM.Interface.cifar10" and os.environ.get("MNSIM_ROUND_CIFAR10_ROOT"):
        return "round_orchestrated_cifar10"
    return module


def _output_open_failure(
    request: RoundOrchestratedRunRequest, error: Exception
) -> RoundOrchestratedRunResult:
    requested_stages = request.requested_stages()
    return RoundOrchestratedRunResult(
        status="failed",
        exit_code=2,
        request=request,
        stage_statuses={
            stage_name: StageStatus(name=stage_name, status="not_run")
            for stage_name in requested_stages
        },
        errors=[
            ErrorRecord(
                code=INVALID_OUTPUT_DIRECTORY,
                detail=_exception_detail(error),
            )
        ],
    )


def _validate_inputs(
    request: RoundOrchestratedRunRequest, requested_stages: tuple[str, ...]
) -> ErrorRecord | None:
    if not requested_stages:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="at least one stage must be requested",
        )
    if request.schedule_only and request.infer_with_schedule:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="schedule_only and infer_with_schedule are mutually exclusive",
        )
    if (request.schedule_only or request.infer_with_schedule) and not request.run_hardware:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="schedule generation and scheduled inference require run_hardware",
        )
    if request.schedule_only and request.schedule_plan_out is None:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="schedule_only requires schedule_plan_out",
        )
    if request.infer_with_schedule and request.schedule_plan_in is None:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="infer_with_schedule requires schedule_plan_in",
        )
    if request.infer_with_schedule and request.custom_schedule_in is not None:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="infer_with_schedule and custom_schedule_in are mutually exclusive",
        )
    if request.custom_schedule_in is not None and not request.run_hardware:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="custom_schedule_in requires run_hardware",
        )
    if request.schedule_policy not in {
        "seq",
        "flashsef_ptr",
        "flashsef-ptr",
        "snn_flashsef_ptr",
        "snn-flashsef-ptr",
        "snn_native_temporal",
        "snn_flashsef_temporal",
        "snn-flashsef-temporal",
    }:
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail=(
                "schedule_policy must be seq, flashsef_ptr, snn_flashsef_ptr, "
                "snn_native_temporal, or snn_flashsef_temporal"
            ),
        )
    if "accuracy" in requested_stages and not (request.run_original or request.run_pim):
        return ErrorRecord(
            code=INVALID_STAGE_SELECTION,
            detail="at least one accuracy path must be enabled",
        )
    if type(request.num_classes) is not int or request.num_classes <= 0:
        return ErrorRecord(
            code=INVALID_NUM_CLASSES,
            detail="num_classes must be a positive integer",
        )
    if request.accuracy_seed is not None and (
        type(request.accuracy_seed) is not int
        or request.accuracy_seed < 0
        or request.accuracy_seed > 2**32 - 1
    ):
        return ErrorRecord(
            code=INVALID_ACCURACY_SEED,
            detail="accuracy_seed must be an integer in [0, 4294967295]",
        )
    if (
        "snn" in request.network.lower()
        and request.run_pim
        and request.accuracy_seed not in {None, 112}
    ):
        return ErrorRecord(
            code=INVALID_ACCURACY_SEED,
            detail="formal SNN PIM accuracy seed is fixed to 112",
        )
    if request.max_samples is not None and (
        type(request.max_samples) is not int or request.max_samples <= 0
    ):
        return ErrorRecord(code="INVALID_MAX_SAMPLES", detail="max_samples must be positive")
    if "accuracy" in requested_stages and (
        not isinstance(request.dataset_module, str)
        or not request.dataset_module.strip()
    ):
        return ErrorRecord(
            code=DATASET_MODULE_REQUIRED,
            detail="dataset_module is required when accuracy is requested",
        )
    if not _is_readable_regular_file(request.weights):
        return ErrorRecord(
            code=CHECKPOINT_LOAD_FAILED,
            detail=f"weights must be a readable regular file: {request.weights}",
        )
    if not _is_readable_regular_file(request.sim_config):
        return ErrorRecord(
            code=PHASE1_INVALID,
            detail=f"sim_config must be a readable regular file: {request.sim_config}",
        )
    if request.infer_with_schedule and not _is_readable_regular_file(request.schedule_plan_in):
        return ErrorRecord(
            code=PHASE1_INVALID,
            detail=f"schedule_plan_in must be a readable regular file: {request.schedule_plan_in}",
        )
    if request.custom_schedule_in is not None and not _is_readable_regular_file(
        request.custom_schedule_in
    ):
        return ErrorRecord(
            code=PHASE1_INVALID,
            detail=(
                "custom_schedule_in must be a readable regular file: "
                f"{request.custom_schedule_in}"
            ),
        )
    if request.schedule_only and request.schedule_plan_out is not None:
        try:
            Path(request.schedule_plan_out).parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return ErrorRecord(code=INVALID_OUTPUT_DIRECTORY, detail=str(error))
    return None


def _is_readable_regular_file(path_like: Path) -> bool:
    path = Path(path_like)
    try:
        if not path.is_file() or not os.access(path, os.R_OK):
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def _context_error_code(error: Exception, deps: RunnerDependencies) -> str:
    if isinstance(error, RunnerError) and error.code == CHECKPOINT_LOAD_FAILED:
        return CHECKPOINT_LOAD_FAILED
    if isinstance(error, PrequantizedWeightError):
        return CHECKPOINT_LOAD_FAILED
    if _is_checkpoint_error(error, deps):
        return CHECKPOINT_LOAD_FAILED
    return PHASE1_INVALID


def _is_checkpoint_error(error: BaseException, deps: RunnerDependencies) -> bool:
    predicate = deps.checkpoint_error_predicate
    for candidate in _exception_chain(error):
        if isinstance(candidate, deps.checkpoint_error_types):
            return True
        if _is_known_torch_checkpoint_runtime_error(candidate):
            return True
        if predicate is not None and predicate(candidate):
            return True
    return False


def _is_known_torch_checkpoint_runtime_error(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and any(
        marker in str(error).casefold()
        for marker in _TORCH_CHECKPOINT_RUNTIME_MARKERS
    )


def _exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None


def _run_accuracy_stage(
    store: ArtifactStore,
    result: RoundOrchestratedRunResult,
    request: RoundOrchestratedRunRequest,
    original_context: Any,
    pim_context: Any,
    deps: RunnerDependencies,
    *,
    deployment_manifest: dict[str, Any] | None = None,
) -> None:
    _set_stage_status(result, "accuracy", "running")
    _write_manifest(store, result)
    try:
        accuracy_report = _evaluate_accuracy(
            request,
            original_context,
            pim_context,
            deps,
            deployment_manifest=deployment_manifest,
            persist_readc_artifact=lambda payload: _write_artifact(
                store,
                result,
                "readc_calibration.json",
                payload,
            ),
        )
    except Exception as error:
        stage_error = ErrorRecord(
            code=ACCURACY_FAILED,
            detail=_exception_detail(error),
            stage="accuracy",
        )
        _write_artifact(
            store,
            result,
            "accuracy_report.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error": stage_error.to_dict(),
            },
        )
        _record_stage_error(result, "accuracy", stage_error)
    else:
        try:
            _write_artifact(store, result, "accuracy_report.json", accuracy_report)
        except (TypeError, ValueError) as error:
            stage_error = ErrorRecord(
                code=ACCURACY_FAILED,
                detail=_exception_detail(error),
                stage="accuracy",
            )
            _write_artifact(
                store,
                result,
                "accuracy_report.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "error": stage_error.to_dict(),
                },
            )
            _record_stage_error(result, "accuracy", stage_error)
        else:
            result.accuracy_report = accuracy_report
            summary = {}
            original_label, pim_label = _accuracy_report_labels(request.network)
            if accuracy_report.get(original_label) is not None:
                summary[original_label] = accuracy_report[original_label]
            if accuracy_report.get(pim_label) is not None:
                summary[pim_label] = accuracy_report[pim_label]
            _set_stage_status(
                result,
                "accuracy",
                "succeeded",
                summary=summary,
            )
    _write_manifest(store, result)


def _evaluate_accuracy(
    request: RoundOrchestratedRunRequest,
    original_context: Any,
    pim_context: Any = None,
    deps: RunnerDependencies | None = None,
    *,
    deployment_manifest: dict[str, Any] | None = None,
    persist_readc_artifact: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if deps is None:
        deps = pim_context
        pim_context = original_context
    if not isinstance(deps, RunnerDependencies):
        raise TypeError("_evaluate_accuracy requires RunnerDependencies")
    is_snn = "snn" in request.network.lower()
    accuracy_context = pim_context if request.run_pim else original_context
    snn_accuracy_config = getattr(accuracy_context, "snn_accuracy_config", None)
    profile = (
        getattr(snn_accuracy_config, "hardware_profile", None)
        if is_snn
        else None
    )
    if is_snn and profile is None:
        # Compatibility for injected legacy test contexts. Public contexts
        # always obtain this value from SimConfig in TrainTestInterface.
        profile = ADC_DIGITAL_LIF
    adc_action = "FIX" if request.fixed_quantization_range else "SCALE"
    adc_mode = str(getattr(accuracy_context, "hardware_config", {}).get(
        "ADC_quantization_mode", "UNIFORM"
    )).upper()
    if adc_mode in {"ADAPTIVE", "STATIC_NONUNIFORM", "READC_LLOYD_MAX"}:
        adc_action = "ADAPTIVE"
    if is_snn and request.run_pim:
        adc_action = "ADAPTIVE" if profile == ADC_DIGITAL_LIF else None
    net_bits = _quiet_call(pim_context.get_net_bits) if request.run_pim else None
    updated_bits = None
    is_formal_snn_pim = is_snn and request.run_pim
    device_contract = None
    source_checkpoint_sha256 = file_sha256(request.weights) if is_snn else None
    checkpoint_sha256 = (
        deployment_manifest["deployment_checkpoint_sha256"]
        if deployment_manifest is not None
        else source_checkpoint_sha256
    )
    deployment_artifact_checksum = None
    weight_mapping_sha256 = None
    readc_artifact = None
    lif_artifact = None
    lif_perturbation = None
    calibration_split_identity = None
    test_split_identity = None
    if is_snn:
        calibration_identity_reader = getattr(
            pim_context, "calibration_split_identity", None
        )
        if callable(calibration_identity_reader):
            calibration_split_identity = _quiet_call(calibration_identity_reader)
            _validate_snn_accuracy_split_identity(
                calibration_split_identity, role="calibration"
            )
        test_identity_reader = getattr(
            accuracy_context, "test_split_identity", None
        )
        if callable(test_identity_reader):
            test_split_identity = _quiet_call(test_identity_reader)
            _validate_snn_accuracy_split_identity(
                test_split_identity, role="test"
            )
    if request.run_pim:
        deployment_artifact_checksum = _bit_weights_checksum(
            net_bits,
            allow_opaque=deps.interface_factory is not TrainTestInterface,
        )
        if is_formal_snn_pim:
            device_contract = _load_accuracy_device_contract(request.sim_config)
            accuracy_rng = np.random.default_rng(device_contract.device_seed)
            updated_bits = _quiet_call(
                deps.weight_update,
                str(request.sim_config),
                net_bits,
                is_SAF=0,
                is_Variation=1,
                is_Rratio=1,
                rng=accuracy_rng,
            )
            weight_mapping_sha256 = _bit_weights_checksum(
                updated_bits,
                allow_opaque=deps.interface_factory is not TrainTestInterface,
            )
            if profile == ADC_DIGITAL_LIF and (
                snn_accuracy_config is not None or adc_mode == "READC_LLOYD_MAX"
            ):
                identity = ReADCArtifactIdentity(
                    checkpoint_sha256=checkpoint_sha256,
                    dataset_split_identity=(
                        calibration_split_identity
                        if calibration_split_identity is not None
                        else _quiet_call(pim_context.calibration_split_identity)
                    ),
                    device_model_id=device_contract.model_id,
                    device_seed=device_contract.device_seed,
                    weight_mapping_sha256=weight_mapping_sha256,
                    adc_precision=5,
                    percentile_low=0.5,
                    percentile_high=99.5,
                    max_samples_per_layer=8000,
                )
                calibration_split_identity = identity.dataset_split_identity
                readc_artifact = _quiet_call(
                    pim_context.calibrate_readc,
                    updated_bits,
                    identity=identity,
                    max_samples=request.max_samples,
                )
                if persist_readc_artifact is None:
                    raise RuntimeError(
                        "formal ReADC calibration requires durable artifact persistence"
                    )
                persist_readc_artifact(readc_artifact.to_dict())
                _quiet_call(
                    pim_context.install_readc_artifact,
                    readc_artifact,
                    expected_identity=identity,
                )
            elif profile == FRONTEND_LUT_LIF_LUT:
                lif_artifact = getattr(
                    pim_context, "lif_perturbation_artifact", None
                )
                perturbation_mode = str(
                    getattr(
                        snn_accuracy_config,
                        "lif_perturbation_mode",
                        "qualified_artifact",
                    )
                )
                if lif_artifact is None and perturbation_mode != "none":
                    raise ValueError(
                        "frontend_lut_lif_lut requires a qualified "
                        "LIF perturbation artifact"
                    )
                if lif_artifact is not None:
                    lif_perturbation = LIFPerturbationRuntime(
                        lif_artifact,
                        seed=snn_accuracy_config.lif_perturbation_seed,
                    )
            elif profile != ADC_DIGITAL_LIF:
                raise ValueError(f"unsupported SNN hardware profile: {profile}")
        else:
            with _temporary_accuracy_seed(request.accuracy_seed):
                updated_bits = _quiet_call(
                    deps.weight_update,
                    str(request.sim_config),
                    net_bits,
                    is_SAF=int(request.enable_saf),
                    is_Variation=int(request.enable_variation),
                    is_Rratio=int(request.enable_r_ratio),
                )
                if adc_mode == "READC_LLOYD_MAX":
                    raise ValueError(
                        "READC_LLOYD_MAX is only supported by the formal SNN "
                        "adc_digital_lif accuracy path"
                    )
    eval_kwargs = {"adc_action": adc_action}
    if request.max_samples is not None:
        eval_kwargs["max_samples"] = request.max_samples
    original_accuracy = None
    if request.run_original:
        original_method = (
            "TRADITION" if "snn" in request.network.lower() else "FIX_TRAIN"
        )
        original_accuracy = _quiet_call(
            original_context.origin_evaluate, method=original_method, **eval_kwargs
        )
        if is_snn and callable(getattr(original_context, "test_split_identity", None)):
            test_split_identity = _quiet_call(original_context.test_split_identity)
        if request.run_pim and original_context is not pim_context:
            _release_context_device(original_context)
    pim_accuracy = None
    if request.run_pim:
        if is_snn and snn_accuracy_config is not None:
            pim_accuracy = _quiet_call(
                pim_context.set_net_bits_evaluate,
                updated_bits,
                adc_action=adc_action,
                pim_profile=profile,
                lif_perturbation=lif_perturbation,
                **({"max_samples": request.max_samples} if request.max_samples is not None else {}),
            )
            if is_snn and test_split_identity is None and callable(
                getattr(pim_context, "test_split_identity", None)
            ):
                test_split_identity = _quiet_call(pim_context.test_split_identity)
        else:
            pim_accuracy = _quiet_call(
                pim_context.set_net_bits_evaluate,
                updated_bits,
                **eval_kwargs,
            )
            if is_snn and test_split_identity is None and callable(
                getattr(pim_context, "test_split_identity", None)
            ):
                test_split_identity = _quiet_call(pim_context.test_split_identity)
    original_label, pim_label = _accuracy_report_labels(request.network)
    report = {
        "schema_version": 1,
        "status": "succeeded",
        original_label: original_accuracy,
        pim_label: pim_accuracy,
        "adc_action": adc_action if request.run_pim else None,
        "controls": (
            {
                "enable_saf": False,
                "enable_variation": True,
                "enable_r_ratio": True,
                "accuracy_seed": 112,
                "source": "final_rram_contract",
            }
            if is_formal_snn_pim
            else {
                "enable_saf": bool(request.enable_saf),
                "enable_variation": bool(request.enable_variation),
                "enable_r_ratio": bool(request.enable_r_ratio),
                "accuracy_seed": request.accuracy_seed,
                "fixed_quantization_range": bool(request.fixed_quantization_range),
            }
        ),
    }
    if is_snn and request.run_original:
        report["snn_original_path"] = _snn_original_path_metadata(original_context)
    if is_snn and test_split_identity is not None:
        report["snn_accuracy_data"] = {
            "source_checkpoint": str(request.weights),
            "source_checkpoint_kind": "software_pretrain",
            "calibration_split": calibration_split_identity,
            "test_split": test_split_identity,
        }
    device_metadata = _device_nonideality_metadata(request.sim_config, request)
    if device_metadata["configured"] and device_contract is not None:
        device_metadata["variation_percent"] = device_contract.variation_percent
        variation_model = getattr(
            device_contract,
            "variation_model",
            "multiplicative_gaussian_resistance",
        )
        programming_error_slope = getattr(
            device_contract, "programming_error_slope", None
        )
        programming_error_intercept_us = getattr(
            device_contract, "programming_error_intercept_us", None
        )
        device_metadata["variation_model"] = variation_model
        if programming_error_slope is not None:
            device_metadata["programming_error"] = {
                "slope": programming_error_slope,
                "intercept_us": programming_error_intercept_us,
                "equation": "sigma_G_uS = max(slope*G_uS + intercept_us, 0)",
            }
        device_metadata["enabled"] = {
            "r_ratio": True,
            "variation": True,
            "saf": False,
        }
    if device_metadata["configured"]:
        report["device_nonidealities"] = device_metadata
    if device_contract is not None:
        report.update(
            {
                "accuracy_device_model": device_contract.model_id,
                "accuracy_device_variation_percent": (
                    device_contract.variation_percent
                ),
                "accuracy_device_variation_model": (
                    getattr(
                        device_contract,
                        "variation_model",
                        "multiplicative_gaussian_resistance",
                    )
                ),
                "accuracy_device_seed": device_contract.device_seed,
                "accuracy_variation_sampling_scope": (
                    "once_per_physical_cell_per_pim_accuracy_run"
                ),
            }
        )
    if adc_mode != "UNIFORM" and (
        not is_snn or profile == ADC_DIGITAL_LIF
    ):
        report["adc_quantization_mode"] = adc_mode
    if readc_artifact is not None:
        layer_checksums = {
            layer_id: table.get("table_checksum")
            for layer_id, table in sorted(readc_artifact.layers.items())
        }
        readc_calibration = {
            **READC_PROVENANCE,
            "calibration_source": "dataset_training_loader",
            "artifact": "readc_calibration.json",
            "artifact_checksum": readc_artifact.artifact_checksum,
            "layer_checksums": layer_checksums,
        }
        report["readc_calibration"] = readc_calibration
        # Retain the established provenance key while the canonical report
        # nests identity-bound checksums under readc_calibration.
        report["adc_model"] = {
            **READC_PROVENANCE,
            "calibration_source": "dataset_training_loader",
            "artifact": "readc_calibration.json",
            "artifact_checksum": readc_artifact.artifact_checksum,
        }
    if is_snn:
        report.update(
            {
                "accuracy_compute_model": "mnsim_bit_sliced_weighted_ops",
                "accuracy_hardware_coupling": "profile_selected_numerical_only",
                "accuracy_artifact_checksum": canonical_json_sha256(
                    {
                        "original_accuracy": original_accuracy,
                        "pim_accuracy": pim_accuracy,
                        "checkpoint_sha256": checkpoint_sha256,
                        "weight_mapping_sha256": weight_mapping_sha256,
                        "hardware_profile": profile,
                        "readc_artifact_checksum": (
                            readc_artifact.artifact_checksum
                            if readc_artifact else None
                        ),
                        "lif_perturbation_checksum": (
                            lif_artifact.artifact_checksum if lif_artifact else None
                        ),
                    }
                ),
                "accuracy_adc_model": (
                    "READC_LLOYD_MAX" if readc_artifact else None
                ),
                "accuracy_lif_model": (
                    "spikingjelly.activation_based.neuron.LIFNode"
                ),
                "accuracy_lif_perturbation_profile": (
                    lif_artifact.profile_id
                    if lif_artifact else (
                        "none"
                        if profile == FRONTEND_LUT_LIF_LUT
                        and str(getattr(
                            snn_accuracy_config,
                            "lif_perturbation_mode",
                            "",
                        )) == "none"
                        else None
                    )
                ),
                "accuracy_lif_perturbation_checksum": (
                    lif_artifact.artifact_checksum if lif_artifact else None
                ),
                "accuracy_lif_perturbation_seed": (
                    112 if lif_artifact else None
                ),
                "deployment_artifact_checksum": deployment_artifact_checksum,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "deployment_checkpoint_sha256": (
                    deployment_manifest.get("deployment_checkpoint_sha256")
                    if deployment_manifest is not None else None
                ),
                "direct_prequantized_weight_mode": bool(
                    deployment_manifest is not None
                ),
                "weight_quantization_count": (
                    deployment_manifest.get("quantization_count")
                    if deployment_manifest is not None else None
                ),
                "deployment_device_code_dtype": (
                    "mnsim_differential_code_arrays"
                ),
                "deployment_device_code_range": [0, 255],
                "accuracy_post_variation_mapping_checksum": (
                    weight_mapping_sha256
                ),
                "binary_spike_domain": {
                    "values": [0, 1],
                    "scale": 1,
                    "signed": False,
                    "input_digit_max": 1,
                    "activation_cycle": 1,
                    "rounding": "forbidden",
                    "clamping": "forbidden",
                },
                "hardware_profile": profile,
            }
        )
    if not (request.run_original and request.run_pim):
        report["evaluated_paths"] = {
            "original": bool(request.run_original),
            "pim": bool(request.run_pim),
        }
    return report


def _validate_snn_accuracy_split_identity(identity: Any, *, role: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"SNN accuracy {role} split identity is missing")
    sample_count = identity.get("sample_count")
    if sample_count is not None and int(sample_count) <= 0:
        raise ValueError(f"SNN accuracy {role} split is empty")
    split = str(identity.get("split", "")).strip().lower()
    resolved_split = str(identity.get("resolved_split", "")).strip().lower()
    if role == "calibration":
        if resolved_split == "finetune":
            raise ValueError(
                "SNN accuracy calibration must use pretrain; finetune is not an accuracy source"
            )
        if resolved_split and resolved_split != "pretrain":
            raise ValueError(
                f"SNN accuracy calibration resolved_split must be pretrain, got {resolved_split!r}"
            )
        if split and split != "train":
            raise ValueError(
                f"SNN accuracy calibration logical split must be train, got {split!r}"
            )
    elif role == "test":
        if split and split != "test":
            raise ValueError(
                f"SNN accuracy test split must be test, got {split!r}"
            )
        if resolved_split and resolved_split != "test":
            raise ValueError(
                f"SNN accuracy test resolved_split must be test, got {resolved_split!r}"
            )


def _bit_weights_checksum(value: Any, *, allow_opaque: bool) -> str | None:
    try:
        return bit_weights_sha256(value)
    except (AttributeError, TypeError, ValueError):
        if not allow_opaque:
            raise
        return None


def _load_accuracy_device_contract(sim_config: str | Path):
    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(sim_config, encoding="UTF-8")
    except (OSError, configparser.Error) as error:
        raise FinalRramContractError(
            f"could not read SimConfig path {sim_config}: {error}"
        ) from error
    if not loaded:
        raise FinalRramContractError(f"could not read SimConfig path {sim_config}")
    return load_snn_accuracy_device_contract(parser, source=str(sim_config))


def _accuracy_report_labels(network: str) -> tuple[str, str]:
    if "snn" in str(network).lower():
        return (
            "SNN Original Accuracy",
            "SNN Processing-in-Memory Accuracy",
        )
    return "Original accuracy", "PIM-based computing accuracy"


def _snn_original_path_metadata(context: Any) -> dict[str, Any]:
    network = getattr(context, "net", None)
    lif_layers = [
        layer
        for layer in getattr(network, "layer_list", [])
        if getattr(layer, "layer_config", {}).get("type") == "lif"
    ]
    if not lif_layers:
        raise ValueError("SNN Original Accuracy Path requires at least one LIF layer")
    lif_parameters = {
        (
            float(layer.layer_config["tau"]),
            float(layer.layer_config["v_th"]),
            float(layer.layer_config["v_reset"]),
        )
        for layer in lif_layers
    }
    if len(lif_parameters) != 1:
        raise ValueError("SNN Original Accuracy Path requires uniform LIF parameters")
    tau, v_threshold, v_reset = lif_parameters.pop()
    return {
        "weighted_backend": "pytorch_float",
        "weight_quantization": "none",
        "activation_quantization": "none",
        "neuron_model": "spikingjelly.activation_based.neuron.LIFNode",
        "spikingjelly_step_mode": "m",
        "readout": "timestep_spike_sum",
        "lif": {
            "tau": tau,
            "v_threshold": v_threshold,
            "v_reset": v_reset,
            "decay_input": True,
            "reset_mode": "hard",
        },
    }


def _release_context_device(context: Any) -> None:
    """Keep dual-checkpoint accuracy runs to one GPU-resident model at a time."""
    network = getattr(context, "net", None)
    move = getattr(network, "to", None)
    if callable(move):
        move(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device_nonideality_metadata(sim_config, request):
    parser = configparser.ConfigParser()
    if not parser.read(sim_config, encoding="UTF-8"):
        raise ValueError(f"cannot read SimConfig: {sim_config}")
    required = ("Device_Resistance", "Device_SAF", "Device_Variation")
    if not parser.has_section("Device level") or any(
        not parser.has_option("Device level", option) for option in required
    ):
        return {
            "configured": False,
            "enabled": {
                "r_ratio": bool(request.enable_r_ratio),
                "variation": bool(request.enable_variation),
                "saf": bool(request.enable_saf),
            },
        }
    resistances = [
        float(value) for value in
        parser.get("Device level", "Device_Resistance").split(",")
    ]
    saf = [
        float(value) for value in
        parser.get("Device level", "Device_SAF").split(",")
    ]
    metadata = {
        "configured": True,
        "mapping_order": ["device_level", "r_ratio", "cell_variation", "saf"],
        "variation_granularity": "independent_physical_cell",
        "differential_arrays_share_errors": False,
        "variation_percent": parser.getfloat("Device level", "Device_Variation"),
        "saf_hrs_percent": saf[0],
        "saf_lrs_percent": saf[1],
        "device_resistance_ohm": resistances,
        "r_ratio_hrs_over_lrs": max(resistances) / min(resistances),
        "enabled": {
            "r_ratio": bool(request.enable_r_ratio),
            "variation": bool(request.enable_variation),
            "saf": bool(request.enable_saf),
        },
    }
    final_model = _final_rram_model_metadata(sim_config)
    if final_model is not None:
        metadata["final_rram_model"] = final_model
    return metadata


def _final_rram_model_metadata(sim_config: str | Path) -> dict[str, str] | None:
    parser = configparser.ConfigParser()
    if not parser.read(sim_config, encoding="UTF-8"):
        raise ValueError(f"cannot read SimConfig: {sim_config}")
    section = "Final RRAM Model"
    if not parser.has_section(section):
        return None
    option_map = {
        "model_id": "Model_ID",
        "model_json": "Model_JSON",
        "source_file": "Source_File",
        "source_sha256": "Source_SHA256",
        "nominal_fit": "Nominal_Fit",
        "variation_estimator": "Variation_Estimator",
        "reset_model": "RESET_Model",
    }
    missing = [option for option in option_map.values() if not parser.has_option(section, option)]
    if missing:
        raise ValueError(
            f"SimConfig [{section}] is missing provenance fields: {', '.join(missing)}"
        )
    return {
        key: parser.get(section, option)
        for key, option in option_map.items()
    }


@contextmanager
def _temporary_accuracy_seed(seed: int | None):
    """Scope device draws and calibration-loader shuffle to one accuracy run."""

    if seed is None:
        yield
        return

    state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        torch.random.set_rng_state(torch_state)


_temporary_numpy_seed = _temporary_accuracy_seed


def _schedule_required_xbars(plan: Any) -> dict[str, int]:
    return {
        str(descriptor.layer_id): int(descriptor.required_xbar_count)
        for descriptor in list(getattr(plan, "layer_resource_descriptors", []) or [])
    }


def _schedule_boundary_view(item: Any) -> BoundaryInputView:
    return BoundaryInputView(
        input_view_id=f"schedule:{item.item_id}",
        item_id=str(item.item_id),
        readable_state_ids=[],
        state_bindings={},
        warnings=[],
    )


def _schedule_xbar_rows(sim_config_path: str | Path) -> int:
    parser = configparser.ConfigParser()
    parser.read(sim_config_path, encoding="UTF-8")
    for section in ("Crossbar level", "Crossbar Level", "Architecture level"):
        if parser.has_option(section, "Xbar_Size"):
            raw = parser.get(section, "Xbar_Size").replace("[", "").replace("]", "")
            try:
                return max(1, int(raw.split(",")[0].strip()))
            except (TypeError, ValueError):
                pass
    return 128


def _schedule_tile_coordinates(sim_config_path: str | Path) -> dict[int, tuple[int, int]]:
    """Resolve MNSIM physical tile ids to the coordinates used by its TCG."""
    parser = configparser.ConfigParser()
    if not parser.read(sim_config_path, encoding="UTF-8"):
        raise ValueError(f"could not read SimConfig for SNN tile topology: {sim_config_path}")
    rows, columns = _sim_config_pair(parser, "Architecture level", "Tile_Num")
    if rows == 0:
        rows = columns = 8
    connection = int(parser.get("Architecture level", "Tile_Connection"))
    from MNSIM.Mapping_Model.Tile_connection_graph import (
        generate_hui_matrix,
        generate_normal_matrix,
        generate_snake_matrix,
        generate_zigzag_matrix,
    )

    generators = {
        0: generate_normal_matrix,
        1: generate_snake_matrix,
        2: generate_hui_matrix,
        3: generate_zigzag_matrix,
    }
    try:
        order = generators[connection](rows, columns)
    except KeyError as error:
        raise ValueError(f"unsupported Tile_Connection value {connection}") from error
    return {
        int(order[row, column]): (row, column)
        for row in range(rows)
        for column in range(columns)
    }


def _schedule_code_vector(
    *,
    fragment: Any,
    weight_codes_by_layer: Mapping[str, Mapping[str, Any]],
    xbar_rows: int,
) -> list[int]:
    layer_codes = weight_codes_by_layer.get(str(fragment.layer_id))
    if not isinstance(layer_codes, Mapping):
        raise Phase2Invalid(
            "PHYSICAL_WRITE_CODES_MISSING",
            f"missing checkpoint device codes for layer {fragment.layer_id}",
        )
    input_block = getattr(fragment, "input_block_index", None)
    if not isinstance(input_block, Integral) or int(input_block) < 0:
        input_block = int(fragment.input_range[0]) // int(xbar_rows)
    input_block = int(input_block)
    plane = int(fragment.weight_plane_index)
    if int(fragment.polarity_count) == 2:
        candidates = [
            f"split{input_block}_weight{plane}_positive",
            f"split{input_block}_weight{plane}_negative",
        ]
    else:
        candidates = [f"split{input_block}_weight{plane}"]
    values: list[int] = []
    start, end = (int(value) for value in fragment.output_range)
    for key in candidates:
        if key not in layer_codes:
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"layer {fragment.layer_id} is missing weight plane {plane} split {input_block}: {key}",
            )
        matrix = np.asarray(layer_codes[key])
        if matrix.ndim < 2 or start < 0 or end <= start or end > matrix.shape[0]:
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"invalid output range {fragment.output_range} for {fragment.layer_id}:{key}",
            )
        values.extend(int(value) for value in matrix[start:end].reshape(-1).tolist())
    return values


def _schedule_physical_target_entries(
    *,
    fragment: Any,
    group_slot: Mapping[str, Any],
    weight_codes_by_layer: Mapping[str, Mapping[str, Any]],
    xbar_rows: int,
) -> list[dict[str, Any]]:
    """Build compact matrices keyed by the physical fragment slots.

    The matrices retain the fragment's true input/output shape. The native
    scheduler pads them to the configured xbar only while scoring a write, so
    an absent fragment slot can remain resident exactly as it does in Phase 2.
    """

    layer_codes = weight_codes_by_layer.get(str(fragment.layer_id))
    if not isinstance(layer_codes, Mapping):
        raise Phase2Invalid(
            "PHYSICAL_WRITE_CODES_MISSING",
            f"missing checkpoint device codes for layer {fragment.layer_id}",
        )
    input_block = getattr(fragment, "input_block_index", None)
    if not isinstance(input_block, Integral) or int(input_block) < 0:
        input_block = int(fragment.input_range[0]) // int(xbar_rows)
    input_block = int(input_block)
    plane = int(fragment.weight_plane_index)
    if int(fragment.polarity_count) == 2:
        candidates = [
            f"split{input_block}_weight{plane}_positive",
            f"split{input_block}_weight{plane}_negative",
        ]
    else:
        candidates = [f"split{input_block}_weight{plane}"]
    start, end = (int(value) for value in fragment.output_range)
    entries: list[dict[str, Any]] = []
    for polarity_index, key in enumerate(candidates):
        if key not in layer_codes:
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"layer {fragment.layer_id} is missing weight plane {plane} split {input_block}: {key}",
            )
        array = np.asarray(layer_codes[key])
        if array.ndim < 2 or start < 0 or end <= start or end > array.shape[0]:
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"invalid output range {fragment.output_range} for {fragment.layer_id}:{key}",
            )
        output_major = array[start:end].reshape(end - start, -1)
        matrix = np.asarray(output_major.T, dtype=np.int32)
        entries.append(
            {
                "slot": [
                    int(group_slot["pe_id"]),
                    int(group_slot["group_id"]),
                    int(polarity_index),
                ],
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "values": [int(value) for value in matrix.reshape(-1).tolist()],
            }
        )
    return entries


def _schedule_round_payload(
    placement: Any,
    weight_codes_by_layer: Mapping[str, Mapping[str, Any]],
    sim_config_path: str | Path,
) -> tuple[tuple[LogicalTile, ...], tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    physical_targets: dict[int, list[dict[str, Any]]] = {}
    slots = getattr(placement, "fragment_slots_by_id", {})
    fragments = getattr(placement, "fragment_specs_by_id", {})
    rows = _schedule_xbar_rows(sim_config_path)
    for fragment_id in sorted(fragments):
        fragment = fragments[fragment_id]
        tile_id = int(slots[fragment_id]["tile_id"])
        grouped.setdefault(tile_id, []).extend(
            _schedule_code_vector(
                fragment=fragment,
                weight_codes_by_layer=weight_codes_by_layer,
                xbar_rows=rows,
            )
        )
        physical_targets.setdefault(tile_id, []).extend(
            _schedule_physical_target_entries(
                fragment=fragment,
                group_slot=slots[fragment_id],
                weight_codes_by_layer=weight_codes_by_layer,
                xbar_rows=rows,
            )
        )
    tiles = tuple(
        LogicalTile(
            tile_id=f"tile:{tile_id}",
            target_state=tuple(values) or (0,),
            local_slots={"tile_id": tile_id},
            metadata={"_native_physical_targets": physical_targets[tile_id]},
        )
        for tile_id, values in sorted(grouped.items())
    )
    return tiles, tuple(sorted(grouped))


def _remap_schedule_placement(
    snapshot: Mapping[str, Any],
    assignment: Mapping[str, int],
    policy_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    remapped = deepcopy(dict(snapshot))
    slots = remapped.get("fragment_slots_by_id", {})
    slot_permutations = dict((policy_metadata or {}).get("slot_permutations", {}) or {})
    for slot in slots.values():
        source = int(slot["tile_id"])
        permutation = slot_permutations.get(f"tile:{source}", {})
        local_slot = f"{int(slot['pe_id'])}:{int(slot['group_id'])}"
        destination_local = permutation.get(local_slot)
        if isinstance(destination_local, str) and ":" in destination_local:
            destination_pe, destination_group = destination_local.split(":", 1)
            slot["pe_id"] = int(destination_pe)
            slot["group_id"] = int(destination_group)
        target = assignment.get(f"tile:{source}")
        if target is not None:
            slot["tile_id"] = int(target)
    tcg = remapped.get("tcg")
    if isinstance(tcg, dict):
        remapped["tcg"] = deepcopy(tcg)
    return remapped


def _build_weight_schedule_artifact(
    *,
    plan: Any,
    model_ir: Any,
    context: Any,
    request: RoundOrchestratedRunRequest,
) -> Any:
    custom_schedule = (
        load_custom_schedule(request.custom_schedule_in)
        if request.custom_schedule_in is not None
        else None
    )
    required_xbars = _schedule_required_xbars(plan)
    lowerer = RoundLowerer(
        xbar_size=tuple(plan.hardware_budget["xbar_size"]),
        sim_config_path=str(request.sim_config),
        required_xbars_by_layer=required_xbars,
        latency_api="nopipe" if request.disable_inner_pipeline else "pipe",
    )
    codes = _weight_codes_by_layer(context, model_ir)
    logical_rounds: list[RoundSchedulingContext] = []
    snapshots: dict[str, dict[str, Any]] = {}
    physical_ids: set[int] = set()
    weighted_item_ids = {
        str(item.item_id)
        for item in list(getattr(plan, "execution_items", []) or [])
        if isinstance(item, WeightedPhysicalRound)
    }
    reorder_block = -1
    for item in list(getattr(plan, "execution_items", []) or []):
        if not isinstance(item, WeightedPhysicalRound):
            reorder_block += 1
            continue
        placement = lowerer.prepare_round(item, _schedule_boundary_view(item)).placement
        if placement is None:
            raise Phase2Invalid("MISSING_ROUND_PLACEMENT", f"{item.item_id} has no placement")
        snapshot = serialize_round_placement(placement)
        tiles, tile_ids = _schedule_round_payload(placement, codes, request.sim_config)
        physical_ids.update(tile_ids)
        snapshots[str(item.item_id)] = snapshot
        logical_rounds.append(
            RoundSchedulingContext(
                round_id=str(item.item_id),
                round_index=len(logical_rounds),
                logical_tiles=tiles,
                metadata={
                    "placement_snapshot": snapshot,
                    "reorder_block": reorder_block,
                    "predecessor_round_ids": sorted(
                        {
                            str(ref.producer_item_id)
                            for ref in list(getattr(item, "external_inputs", []) or [])
                            if ref.producer_item_id in weighted_item_ids
                        }
                    ),
                },
            )
        )
    if not physical_ids:
        physical_ids = set(range(int(plan.hardware_budget.get("tile_nums", 1))))
    native_policy = request.schedule_policy in {
        "snn_flashsef_ptr",
        "snn-flashsef-ptr",
        "snn_native_temporal",
        "snn_flashsef_temporal",
        "snn-flashsef-temporal",
        "snn-native-temporal",
    }
    device = (
        load_native_device_model_from_sim_config(str(request.sim_config))
        if native_policy
        else load_device_model_from_sim_config(str(request.sim_config))
    )
    problem = build_problem_from_round_plan(
        plan,
        logical_rounds=logical_rounds,
        constraints=SchedulingConstraints(tuple(sorted(physical_ids))),
        device_model=device,
        model_ir=model_ir,
        weights_snapshot={"checkpoint_sha256": file_sha256(request.weights)},
        sim_config_sha256=file_sha256(request.sim_config),
        metadata={
            "round_plan_snapshot_sha256": canonical_sha256(plan),
            "backend_kind": str(getattr(model_ir, "backend_kind", "")),
        },
    )
    if custom_schedule is not None:
        policy = None
    elif request.schedule_policy in {"snn_flashsef_ptr", "snn-flashsef-ptr"}:
        if str(getattr(model_ir, "backend_kind", "")).lower() != "snn":
            raise SNNActivityProfileError(
                "snn_flashsef_ptr requires a ModelIR declared with backend_kind='snn'"
            )
        policy = SNNFlashSeFPTRPolicy()
    elif request.schedule_policy in {"flashsef_ptr", "flashsef-ptr"}:
        policy = FlashSeFPTRPolicy()
    elif native_policy:
        if str(getattr(model_ir, "backend_kind", "")).lower() != "snn":
            raise SNNActivityProfileError(
                "snn_native_temporal requires a ModelIR declared with backend_kind='snn'"
            )
        if request.schedule_policy in {
            "snn_flashsef_temporal",
            "snn-flashsef-temporal",
        }:
            from MNSIM.Scheduling.weight_scheduling.policies.snn_flashsef_temporal import (
                SNNFlashSeFTemporalPolicy,
            )

            policy = SNNFlashSeFTemporalPolicy()
        else:
            policy = SNNNativeTemporalPolicy()
    else:
        policy = SequentialPolicy()
    if custom_schedule is not None:
        artifact = generate_weight_schedule_from_custom(problem, custom_schedule)
    else:
        artifact = generate_weight_schedule(problem, policy)
    if request.custom_schedule_in is not None or native_policy:
        _apply_native_round_order(plan, artifact.policy_metadata.get("round_order", []))
    plan_hash = canonical_sha256(plan)
    records = []
    for record in artifact.rounds:
        snapshot = record.placement_snapshot or snapshots[record.round_id]
        remapped = remap_custom_schedule_placement(
            snapshot, record.logical_to_physical_tile
        ) if custom_schedule is not None else _remap_schedule_placement(
            snapshot,
            record.logical_to_physical_tile,
            record.policy_metadata,
        )
        records.append(replace(record, placement_snapshot=remapped))
    return replace(
        artifact,
        schema_version=2,
        rounds=records,
        round_plan_snapshot=plan.to_dict(),
        base_round_plan_sha256=plan_hash,
        policy_metadata={
            **artifact.policy_metadata,
            "base_round_plan_sha256": plan_hash,
            "round_plan_snapshot_sha256": plan_hash,
            "schedule_policy": artifact.policy_id,
        },
    )


def _apply_native_round_order(plan: Any, round_order: Any) -> bool:
    """Apply only the scheduler's contiguous, dependency-checked round blocks."""

    if not isinstance(round_order, list):
        return False
    weighted_ids = [
        str(item.item_id)
        for item in list(getattr(plan, "execution_items", []) or [])
        if isinstance(item, WeightedPhysicalRound)
    ]
    if set(str(value) for value in round_order) != set(weighted_ids):
        raise Phase1Invalid(
            "INVALID_NATIVE_ROUND_ORDER",
            "native schedule round_order does not match weighted RoundPlan items",
        )
    desired = [str(value) for value in round_order]
    position = 0
    items = list(getattr(plan, "execution_items", []) or [])
    changed = False
    while position < len(items):
        if not isinstance(items[position], WeightedPhysicalRound):
            position += 1
            continue
        end = position
        while end < len(items) and isinstance(items[end], WeightedPhysicalRound):
            end += 1
        block = items[position:end]
        block_ids = [str(item.item_id) for item in block]
        selected = [item_id for item_id in desired if item_id in set(block_ids)]
        if selected != block_ids:
            by_id = {str(item.item_id): item for item in block}
            items[position:end] = [by_id[item_id] for item_id in selected]
            changed = True
        position = end
    for item_index, item in enumerate(items):
        item.item_index = item_index
    plan.execution_items = items
    validate_round_plan(plan)
    return changed


def _run_hardware_stage(
    store: ArtifactStore,
    result: RoundOrchestratedRunResult,
    request: RoundOrchestratedRunRequest,
    context: Any,
    deps: RunnerDependencies,
) -> bool:
    """Return whether hardware failed before a RoundPlan was constructed."""
    _set_stage_status(result, "hardware", "running")
    _write_manifest(store, result)
    try:
        frontend_kwargs = _frontend_sequence_kwargs(context.net)
        model_ir = _quiet_call(
            deps.model_ir_builder,
            context.net,
            model_id=request.network,
            **frontend_kwargs,
        )
        _assert_supported_backend(model_ir)
        counts = count_model_operations(model_ir)
        phase1_projection = _quiet_call(
            deps.phase1_projection_builder, str(request.sim_config)
        )
        model_inputs = _model_inputs_from_root_layers(model_ir)
        if request.infer_with_schedule:
            artifact = load_schedule_artifact(request.schedule_plan_in)
            if not artifact.execution_compatible:
                reason = artifact.migration_metadata.get(
                    "incompatibility_reason",
                    "legacy migration does not contain a current RoundPlan and placement snapshots",
                )
                raise ValueError(f"schedule artifact is audit-only: {reason}")
            if artifact.round_plan_snapshot is None:
                raise ValueError("schedule artifact does not contain a round plan snapshot")
            plan = round_plan_from_dict(artifact.round_plan_snapshot)
            # RoundPlan deserialization normalizes legacy fields; validate the
            # identity against the exact persisted snapshot instead of a
            # lossy re-encoding of the reconstructed object.
            if artifact.base_round_plan_sha256 != canonical_sha256(artifact.round_plan_snapshot):
                raise ValueError("schedule artifact base RoundPlan hash mismatch")
            if artifact.model_ir_sha256 and artifact.model_ir_sha256 != canonical_sha256(model_ir):
                raise ValueError("schedule artifact ModelIR hash mismatch")
            if artifact.sim_config_sha256 and artifact.sim_config_sha256 != file_sha256(request.sim_config):
                raise ValueError("schedule artifact SimConfig hash mismatch")
            expected_weights_hash = canonical_sha256(
                {"checkpoint_sha256": file_sha256(request.weights)}
            )
            if artifact.weights_sha256 and artifact.weights_sha256 != expected_weights_hash:
                raise ValueError("schedule artifact weights hash mismatch")
        else:
            plan = _quiet_call(deps.plan_builder, model_ir, phase1_projection)
        result.round_plan = plan
    except Exception as error:
        _record_stage_error(result, "hardware", _phase1_error_record(error))
        _write_manifest(store, result)
        return True

    _write_artifact(
        store,
        result,
        "round_plan.json",
        {
            "schema_version": 1,
            "status": "running",
            "round_plan": plan.to_dict(),
        },
    )

    if request.schedule_only:
        plan_error = _infeasible_plan_error(plan)
        if plan_error is not None:
            _write_artifact(
                store,
                result,
                "round_plan.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "round_plan": plan.to_dict(),
                    "error": plan_error.to_dict(),
                },
            )
            _record_stage_error(result, "hardware", plan_error)
            _write_manifest(store, result)
            return False
        try:
            artifact = _build_weight_schedule_artifact(
                plan=plan,
                model_ir=model_ir,
                context=context,
                request=request,
            )
            schedule_path = Path(request.schedule_plan_out)
            save_schedule_artifact(schedule_path, artifact)
            _write_artifact(
                store,
                result,
                "round_plan.json",
                {
                    "schema_version": 1,
                    "status": "succeeded",
                    "round_plan": plan.to_dict(),
                },
            )
            _register_existing_artifact(result, "schedule_artifact", schedule_path)
            result.artifacts["schedule_artifact_sha256"] = artifact.sha256()
            _set_stage_status(
                result,
                "hardware",
                "succeeded",
                summary={
                    "execution_mode": "schedule-only",
                    "schedule_artifact": str(schedule_path),
                    "policy_id": artifact.policy_id,
                    "round_count": len(artifact.rounds),
                },
            )
            _write_manifest(store, result)
            return False
        except Exception as error:
            stage_error = _phase1_error_record(error)
            _record_stage_error(result, "hardware", stage_error)
            _write_manifest(store, result)
            return True
    _write_manifest(store, result)
    plan_error = _infeasible_plan_error(plan)
    if plan_error is not None:
        _write_artifact(
            store,
            result,
            "round_plan.json",
            {
                "schema_version": 1,
                "status": "failed",
                "round_plan": plan.to_dict(),
                "error": plan_error.to_dict(),
            },
        )
        _record_stage_error(result, "hardware", plan_error)
        _write_manifest(store, result)
        return False
    custom_artifact = None
    if request.custom_schedule_in is not None:
        try:
            custom_artifact = _build_weight_schedule_artifact(
                plan=plan, model_ir=model_ir, context=context, request=request,
            )
            schedule_path = request.output_dir / "custom.schedule.artifact.json"
            save_schedule_artifact(schedule_path, custom_artifact)
            _register_existing_artifact(result, "schedule_artifact", schedule_path)
            result.artifacts["schedule_artifact_sha256"] = custom_artifact.sha256()
        except Exception as error:
            _record_stage_error(result, "hardware", _phase1_error_record(error))
            _write_manifest(store, result)
            return True
    _write_artifact(
        store,
        result,
        "round_plan.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "round_plan": plan.to_dict(),
        },
    )

    try:
        schedule_artifact = (
            load_schedule_artifact(request.schedule_plan_in)
            if request.infer_with_schedule
            else custom_artifact
        )
        report = _quiet_call(
            deps.phase2_executor,
            plan=plan,
            sim_config_path=str(request.sim_config),
            ops_total=counts.ops_total,
            ops_count_policy=counts.ops_count_policy,
            weight_codes_by_layer=_weight_codes_by_layer(context, model_ir),
            static_backend=deps.static_backend,
            model_inputs=model_inputs,
            latency_api="nopipe" if request.disable_inner_pipeline else "pipe",
            schedule_artifact=schedule_artifact,
        )
        hardware_summary = derive_hardware_rates(report.inference_metrics, counts)
        hardware_summary["sequence_length"] = int(
            getattr(model_ir.sequence_spec, "sequence_length", 1)
        )
        hardware_summary["backend_kind"] = str(model_ir.backend_kind)
        final_model = _final_rram_model_metadata(request.sim_config)
        if final_model is not None:
            hardware_summary["final_rram_model"] = final_model
    except Exception as error:
        stage_error = _phase2_error_record(error)
        _write_artifact(
            store,
            result,
            "phase2_report.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error": stage_error.to_dict(),
            },
        )
        _record_stage_error(result, "hardware", stage_error)
        _write_manifest(store, result)
        return False

    try:
        _write_artifact(
            store,
            result,
            "phase2_report.json",
            {
                "schema_version": 1,
                "status": "succeeded",
                "phase2_report": report.to_dict(),
                "hardware_summary": hardware_summary,
            },
        )
    except (TypeError, ValueError) as error:
        stage_error = ErrorRecord(
            code=PHASE2_FAILED,
            detail=_exception_detail(error),
            stage="hardware",
        )
        _write_artifact(
            store,
            result,
            "phase2_report.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error": stage_error.to_dict(),
            },
        )
        _record_stage_error(result, "hardware", stage_error)
        _write_manifest(store, result)
        return False

    result.phase2_report = report
    result.hardware_summary = hardware_summary
    _set_stage_status(result, "hardware", "succeeded", summary=hardware_summary)
    _write_manifest(store, result)
    return False


def _weight_codes_by_layer(context: Any, model_ir: Any) -> dict[str, Mapping[str, Any]]:
    """Bind real quantized checkpoint codes to canonical Phase-1 layer ids."""

    net = getattr(context, "net", None)
    get_weights = getattr(net, "get_weights", None)
    if not callable(get_weights):
        raise Phase2Invalid(
            "PHYSICAL_WRITE_CODES_MISSING",
            "ANN context network does not expose get_weights()",
        )
    raw_weights = _quiet_call(get_weights)
    if not isinstance(raw_weights, Sequence) or isinstance(
        raw_weights, (str, bytes, bytearray)
    ):
        raise Phase2Invalid(
            "PHYSICAL_WRITE_CODES_MISSING",
            "network get_weights() must return one entry per graph layer",
        )

    result: dict[str, Mapping[str, Any]] = {}
    for layer in getattr(model_ir, "layers", []):
        if not bool(getattr(layer, "is_weighted", False)):
            continue
        attrs = getattr(layer, "backend_attrs", {})
        if not isinstance(attrs, Mapping):
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"layer {getattr(layer, 'layer_id', '<unknown>')} has no MNSIM layer index",
            )
        raw_index = attrs.get("mnsim_layer_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"layer {getattr(layer, 'layer_id', '<unknown>')} has invalid MNSIM layer index",
            )
        index = int(raw_index)
        if index < 0 or index >= len(raw_weights):
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"layer {getattr(layer, 'layer_id', '<unknown>')} has out-of-range MNSIM layer index {index}",
            )
        layer_codes = raw_weights[index]
        if not isinstance(layer_codes, Mapping):
            raise Phase2Invalid(
                "PHYSICAL_WRITE_CODES_MISSING",
                f"weighted layer {getattr(layer, 'layer_id', '<unknown>')} has no checkpoint device-code mapping",
            )
        from MNSIM.Scheduling.depthwise import is_depthwise, expand_depthwise_codes
        if layer.layer_type == "conv" and is_depthwise(layer.shape):
            hardware = net.layer_list[index].hardware_config
            layer_codes = expand_depthwise_codes(
                layer_codes, layer.shape, int(hardware["xbar_size"]), int(hardware["xbar_size"]))
        result[str(layer.layer_id)] = layer_codes
    return result


def _assert_supported_backend(model_ir: Any) -> None:
    backend_kind = str(getattr(model_ir, "backend_kind", ""))
    if backend_kind not in {"ann", "snn"}:
        raise RunnerError(
            UNSUPPORTED_LAYER_TYPE,
            f"round-orchestrated runner supports ann or snn backend_kind, got {backend_kind!r}",
        )
    layer_types = {
        str(getattr(layer, "layer_type", "")).lower()
        for layer in list(getattr(model_ir, "layers", []) or [])
    }
    if backend_kind == "ann" and "lif" in layer_types:
        raise RunnerError(
            UNSUPPORTED_LAYER_TYPE,
            "ann backend_kind cannot contain lif layers; declare an snn sequence_spec",
        )


def _frontend_sequence_kwargs(network: Any) -> dict[str, Any]:
    get_structure = getattr(network, "get_structure", None)
    if not callable(get_structure):
        return {"backend_kind": "ann"}
    structure = _quiet_call(get_structure)
    has_lif = any(
        str(layer.get("type", "")).lower() == "lif"
        for layer in structure
        if isinstance(layer, Mapping)
    )
    if not has_lif:
        return {"backend_kind": "ann"}
    raw_spec = getattr(network, "sequence_spec", None)
    if isinstance(raw_spec, SequenceSpec):
        sequence_spec = raw_spec
    else:
        raw_length = getattr(network, "sequence_length", getattr(network, "T", None))
        if isinstance(raw_length, bool) or not isinstance(raw_length, Integral) or raw_length <= 0:
            raise RunnerError(
                UNSUPPORTED_LAYER_TYPE,
                "SNN network must expose SequenceSpec or positive sequence_length/T",
            )
        recurrent_ids = [
            f"L{index:03d}_lif"
            for index, layer in enumerate(structure)
            if isinstance(layer, Mapping)
            and str(layer.get("type", "")).lower() == "lif"
        ]
        sequence_spec = SequenceSpec(int(raw_length), True, True, recurrent_ids)
    return {"backend_kind": "snn", "sequence_spec": sequence_spec}


def _infeasible_plan_error(plan: Any) -> ErrorRecord | None:
    if bool(getattr(plan, "feasible", False)):
        return None
    detail = getattr(plan, "invalid_reason", None) or "RoundPlan.feasible is false"
    return ErrorRecord(code=PHASE1_INVALID, detail=str(detail), stage="hardware")


def _phase1_error_record(error: Exception) -> ErrorRecord:
    if isinstance(error, RunnerError) and error.code in {
        INVALID_MAC_ESTIMATE,
        UNSUPPORTED_LAYER_TYPE,
    }:
        code = error.code
    elif isinstance(error, Phase1Invalid) and error.code == UNSUPPORTED_LAYER_TYPE:
        code = UNSUPPORTED_LAYER_TYPE
    else:
        code = PHASE1_INVALID
    return ErrorRecord(code=code, detail=_exception_detail(error), stage="hardware")


def _phase2_error_record(error: Exception) -> ErrorRecord:
    if isinstance(error, RunnerError) and error.code == INVALID_MAC_ESTIMATE:
        return ErrorRecord(
            code=INVALID_MAC_ESTIMATE,
            detail=_exception_detail(error),
            stage="hardware",
        )
    return ErrorRecord(code=PHASE2_FAILED, detail=_exception_detail(error), stage="hardware")


def _quiet_call(callable_object: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return callable_object(*args, **kwargs)


def _set_stage_status(
    result: RoundOrchestratedRunResult,
    stage_name: str,
    status: str,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    stage_status = result.stage_statuses[stage_name]
    stage_status.status = status
    stage_status.error = None
    stage_status.summary = summary


def _record_stage_error(
    result: RoundOrchestratedRunResult, stage_name: str, error: ErrorRecord
) -> None:
    stage_status = result.stage_statuses[stage_name]
    stage_status.status = "failed"
    stage_status.error = error
    stage_status.summary = None
    result.errors.append(error)


def _record_top_level_error(
    result: RoundOrchestratedRunResult, error: ErrorRecord
) -> None:
    result.errors.append(error)


def _mark_requested_stages_not_run(result: RoundOrchestratedRunResult) -> None:
    for stage_status in result.stage_statuses.values():
        stage_status.status = "not_run"
        stage_status.error = None
        stage_status.summary = None


def _write_manifest(store: ArtifactStore, result: RoundOrchestratedRunResult) -> None:
    _write_artifact(store, result, "run_manifest.json", result.manifest_dict())


def _write_artifact(
    store: ArtifactStore,
    result: RoundOrchestratedRunResult,
    artifact_name: str,
    payload: dict[str, Any],
) -> None:
    artifact_path = store.write_json(artifact_name, payload)
    result.artifacts[_ARTIFACT_KEYS[artifact_name]] = str(artifact_path)


def _register_existing_artifact(
    result: RoundOrchestratedRunResult,
    artifact_name: str,
    artifact_path: str | Path,
) -> None:
    path = Path(artifact_path)
    if artifact_name not in _ARTIFACT_KEYS:
        raise RunnerError(
            INVALID_OUTPUT_DIRECTORY,
            f"unknown runner artifact: {artifact_name}",
        )
    if not path.is_file():
        raise RunnerError(
            INVALID_OUTPUT_DIRECTORY,
            f"runner artifact was not created: {path}",
        )
    result.artifacts[_ARTIFACT_KEYS[artifact_name]] = str(path)


def _finalize(
    store: ArtifactStore, result: RoundOrchestratedRunResult, *, exit_code: int
) -> RoundOrchestratedRunResult:
    result.status = "succeeded" if exit_code == 0 else "failed"
    result.exit_code = exit_code
    _write_artifact(store, result, "summary.json", result.summary_dict())
    _write_manifest(store, result)
    return result


def _exception_detail(error: BaseException) -> str:
    detail = str(error)
    return detail if detail else type(error).__name__


__all__ = [
    "INVALID_STAGE_SELECTION",
    "INVALID_OUTPUT_DIRECTORY",
    "DATASET_MODULE_REQUIRED",
    "INVALID_NUM_CLASSES",
    "CHECKPOINT_LOAD_FAILED",
    "UNSUPPORTED_LAYER_TYPE",
    "PHASE1_INVALID",
    "PHASE2_FAILED",
    "ACCURACY_FAILED",
    "INVALID_FINAL_RRAM_MODEL",
    "INVALID_MAC_ESTIMATE",
    "RunnerDependencies",
    "run_round_orchestrated_simulation",
]

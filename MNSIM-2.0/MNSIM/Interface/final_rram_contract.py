"""Locked device contracts for bundled round-orchestrated RRAM profiles."""

from __future__ import annotations

import configparser
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


FINAL_RRAM_MODEL_IDS = (
    "rram_stanford_asu_256",
    "rram_memtorch_stanford_pku_256",
    "rram_memtorch_stanford_pku_write256",
)
FINAL_RRAM_MODEL_ALIASES = {
    "stanford_asu": "rram_stanford_asu_256",
    "stanford-asu": "rram_stanford_asu_256",
    "stanford_pku": "rram_memtorch_stanford_pku_256",
    "stanford-pku": "rram_memtorch_stanford_pku_256",
    "stanford_pku_write": "rram_memtorch_stanford_pku_write256",
    **{model_id: model_id for model_id in FINAL_RRAM_MODEL_IDS},
}
FINAL_RRAM_MODEL_SHA256 = {
    "rram_stanford_asu_256": "930a7f3d493e6a3262c4e427e4d1af24091a688323b5abe68a0de0f9fa837740",
    "rram_memtorch_stanford_pku_256": "fffbbb6c67277c050125c91d553b87abaf972707fcad0c75d0e052a649797752",
    "rram_memtorch_stanford_pku_write256": "17d02880abb9fa8282b490e6ee710e8387a24cb7feddb580738e490d643881fd",
}
FINAL_RRAM_DEVICE_SEED = 112


class FinalRramContractError(ValueError):
    """A SimConfig or model artifact violates the locked device contract."""


@dataclass(frozen=True)
class SNNAccuracyDeviceContract:
    model_id: str
    variation_percent: float
    device_seed: int
    enable_r_ratio: bool
    enable_variation: bool
    enable_saf: bool
    saf_percent: tuple[float, float]
    variation_model: str = "multiplicative_gaussian_resistance"
    programming_error_slope: float | None = None
    programming_error_intercept_us: float | None = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_final_rram_model_id(value: str) -> str:
    model_id = FINAL_RRAM_MODEL_ALIASES.get(str(value).strip().lower())
    if model_id is None:
        raise FinalRramContractError(
            f"unsupported device model {value!r}; expected one of "
            f"{list(FINAL_RRAM_MODEL_IDS)!r}"
        )
    return model_id


def final_rram_model_path(
    model_id: str,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    canonical_id = normalize_final_rram_model_id(model_id)
    root = Path(repo_root).resolve() if repo_root is not None else repository_root()
    return root / "device_models" / "final_rram" / "models" / f"{canonical_id}.json"


@lru_cache(maxsize=len(FINAL_RRAM_MODEL_IDS))
def _load_model_cached(model_id: str, model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FinalRramContractError(
            f"cannot read canonical final RRAM model {path}: {exc}"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    expected_digest = FINAL_RRAM_MODEL_SHA256[model_id]
    if digest != expected_digest:
        raise FinalRramContractError(
            f"canonical final RRAM model checksum mismatch for {model_id}: "
            f"expected {expected_digest}, got {digest}"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalRramContractError(
            f"invalid canonical final RRAM model JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or value.get("model_id") != model_id:
        raise FinalRramContractError(
            f"canonical final RRAM model identity mismatch in {path}"
        )
    return value


def load_final_rram_model(
    model_id: str,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    canonical_id = normalize_final_rram_model_id(model_id)
    path = final_rram_model_path(canonical_id, repo_root=repo_root).resolve()
    return copy.deepcopy(_load_model_cached(canonical_id, str(path)))


def apply_final_rram_model(
    parser: configparser.ConfigParser,
    model_id: str,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replace every device-defining field with one canonical model."""

    canonical_id = normalize_final_rram_model_id(model_id)
    model = load_final_rram_model(canonical_id, repo_root=repo_root)
    for section in parser.sections():
        if section.casefold().startswith("device level legacy"):
            parser.remove_section(section)
    device = _mapping(model, "device")
    levels = _mapping(model, "levels")
    source = _mapping(model, "source")
    variation = _mapping(model, "variation")
    nominal_fit = _nominal_fit(model)

    resistance = [float(value) for value in _list(levels, "nominal_resistance_ohm")]
    expected_levels = int(device["levels"])
    if expected_levels < 2 or expected_levels & (expected_levels - 1):
        raise FinalRramContractError(
            f"{canonical_id} levels must be a power of two >= 2, got {expected_levels}"
        )
    if len(resistance) != expected_levels or any(value <= 0 for value in resistance):
        raise FinalRramContractError(
            f"{canonical_id} must contain {expected_levels} positive resistance values"
        )

    capabilities = _capabilities(model)
    hardware_write_supported = _hardware_write_supported(model)
    transition_matrix_model = _has_transition_matrix(model)
    if hardware_write_supported and not transition_matrix_model:
        pulses = [int(value) for value in _list(levels, "pulse")]
        if pulses != list(range(1, expected_levels + 1)):
            raise FinalRramContractError(
                f"{canonical_id} pulse table must be 1..{expected_levels}"
            )
        pulse_width_ns = float(device["pulse_width_ns"])
        program_voltage_v = float(device["programming_voltage_abs_v"])
    elif transition_matrix_model:
        # The canonical JSON transition matrix is consumed directly by the
        # scheduler and Phase 2 ledger.  Legacy scalar pulse fields remain
        # syntactically present in SimConfig but are intentionally neutral.
        pulses = [0] * expected_levels
        pulse_width_ns = 0.0
        program_voltage_v = float(device.get("programming_voltage_abs_v", 0.0))
    else:
        pulses = [0] * expected_levels
        pulse_width_ns = 0.0
        program_voltage_v = 0.0

    if not parser.has_section("Device level"):
        parser.add_section("Device level")
    read_voltage_v = float(device["read_voltage_v"])
    parser.set("Device level", "Read_Voltage", _number(read_voltage_v))
    if transition_matrix_model:
        write_voltage_text = (
            f"{_number(-float(device.get('reset_voltage_abs_v', program_voltage_v)))},"
            f"{_number(program_voltage_v)}"
        )
    else:
        write_voltage_text = f"{_number(program_voltage_v)},{_number(-program_voltage_v)}"
    parser.set(
        "Device level",
        "Write_Voltage",
        write_voltage_text,
    )
    parser.set("Device level", "Write_Latency", _number(pulse_width_ns))
    parser.set(
        "Device level",
        "pulse_latency",
        f"{_number(pulse_width_ns)},{_number(pulse_width_ns)}",
    )
    parser.set("Device level", "SET_pulse", ", ".join(map(str, pulses)))
    parser.set(
        "Device level",
        "RESET_pulse",
        ", ".join(map(str, reversed(pulses))),
    )
    parser.set("Device level", "Device_Level", str(expected_levels))
    parser.set("Device level", "Device_Resistance_Sort", "0")
    parser.set(
        "Device level",
        "Device_Resistance",
        ", ".join(f"{value:.15g}" for value in resistance),
    )
    parser.set(
        "Device level",
        "Device_Variation",
        f"{_variation_percent(variation):.12g}",
    )
    parser.set("Device level", "Device_SAF", "0,0")

    if not parser.has_section("SNN RouteA"):
        parser.add_section("SNN RouteA")
    parser.set("SNN RouteA", "V_read", _number(read_voltage_v))
    parser.set("SNN RouteA", "Linearize_Device_Levels", "0")
    parser.set("SNN RouteA", "Enable_R_Ratio", "1")
    parser.set("SNN RouteA", "Enable_Variation", "1")
    parser.set("SNN RouteA", "Enable_SAF", "0")
    parser.set("SNN RouteA", "Device_Seed", str(FINAL_RRAM_DEVICE_SEED))

    if not parser.has_section("Final RRAM Model"):
        parser.add_section("Final RRAM Model")
    model_reference = (
        Path("device_models")
        / "final_rram"
        / "models"
        / f"{canonical_id}.json"
    ).as_posix()
    parser.set("Final RRAM Model", "Model_ID", canonical_id)
    parser.set("Final RRAM Model", "Model_JSON", model_reference)
    parser.set("Final RRAM Model", "Source_File", _source_file(source))
    parser.set("Final RRAM Model", "Source_SHA256", _source_sha256(source))
    parser.set("Final RRAM Model", "Nominal_Fit", str(nominal_fit["method"]))
    parser.set(
        "Final RRAM Model", "Variation_Estimator", _variation_estimator(variation)
    )
    reset_model = _reset_model(model)
    reset_status = str(reset_model["status"])
    if reset_status == "placeholder":
        reset_status = "mirrored_set_pulse_placeholder"
    parser.set("Final RRAM Model", "RESET_Model", reset_status)
    return _metadata(model, model_reference)


def validate_final_rram_simconfig(
    sim_config: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(sim_config)
    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(path, encoding="UTF-8")
    except (OSError, configparser.Error) as exc:
        raise FinalRramContractError(f"cannot read SimConfig {path}: {exc}") from exc
    if not loaded:
        raise FinalRramContractError(f"cannot read SimConfig {path}")
    return validate_final_rram_parser(parser, source=str(path), repo_root=repo_root)


def validate_final_rram_parser(
    parser: configparser.ConfigParser,
    *,
    source: str = "SimConfig.ini",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if not parser.has_section("Final RRAM Model"):
        raise FinalRramContractError(
            f"{source} is missing [Final RRAM Model]; only "
            f"{list(FINAL_RRAM_MODEL_IDS)!r} are supported"
        )
    try:
        model_id = normalize_final_rram_model_id(
            parser.get("Final RRAM Model", "Model_ID")
        )
    except (configparser.Error, FinalRramContractError) as exc:
        raise FinalRramContractError(
            f"{source} does not identify one supported final RRAM model: {exc}"
        ) from exc
    model = load_final_rram_model(model_id, repo_root=repo_root)
    expected = configparser.ConfigParser()
    expected.add_section("Device level")
    apply_final_rram_model(expected, model_id, repo_root=repo_root)

    _require_exact_int(parser, expected, "Device level", "Device_Level", source)
    _require_exact_int(
        parser, expected, "Device level", "Device_Resistance_Sort", source
    )
    for option in ("Read_Voltage", "Write_Latency", "Device_Variation"):
        _require_close(parser, expected, "Device level", option, source)
    for option in ("Write_Voltage", "pulse_latency", "Device_Resistance"):
        _require_float_vector(parser, expected, "Device level", option, source)
    for option in ("SET_pulse", "RESET_pulse"):
        _require_int_vector(parser, expected, "Device level", option, source)
    _require_float_vector(parser, expected, "Device level", "Device_SAF", source)

    for option in (
        "V_read",
        "Linearize_Device_Levels",
        "Enable_R_Ratio",
        "Enable_Variation",
        "Enable_SAF",
        "Device_Seed",
    ):
        if option == "V_read":
            _require_close(parser, expected, "SNN RouteA", option, source)
        else:
            _require_exact(parser, expected, "SNN RouteA", option, source)

    for option in (
        "Model_ID",
        "Model_JSON",
        "Source_File",
        "Source_SHA256",
        "Nominal_Fit",
        "Variation_Estimator",
        "RESET_Model",
    ):
        _require_exact(parser, expected, "Final RRAM Model", option, source)
    return _metadata(
        model,
        expected.get("Final RRAM Model", "Model_JSON"),
    )


def require_final_rram_hardware_write_support(
    parser: configparser.ConfigParser,
    *,
    source: str = "SimConfig.ini",
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reject accuracy-only profiles before write latency/energy accounting."""

    metadata = validate_final_rram_parser(
        parser,
        source=source,
        repo_root=repo_root,
    )
    if not metadata["hardware_write_accounting_supported"]:
        raise FinalRramContractError(
            f"{metadata['model_id']} does not provide a complete source-to-target "
            "write transition model; hardware write accounting is unavailable"
        )
    return metadata


def load_snn_accuracy_device_contract(
    parser: configparser.ConfigParser,
    *,
    source: str = "SimConfig.ini",
    repo_root: str | Path | None = None,
) -> SNNAccuracyDeviceContract:
    metadata = validate_final_rram_parser(
        parser,
        source=source,
        repo_root=repo_root,
    )
    model = load_final_rram_model(metadata["model_id"], repo_root=repo_root)
    variation = _mapping(model, "variation")
    variation_model = str(
        variation.get("model", "multiplicative_gaussian_resistance")
    )
    programming_error = variation.get("programming_error")
    if programming_error is None:
        programming_error = {}
    if not isinstance(programming_error, dict):
        raise FinalRramContractError(
            "final RRAM variation.programming_error must be an object"
        )
    saf = tuple(
        float(value.strip())
        for value in parser.get("Device level", "Device_SAF").split(",")
    )
    contract = SNNAccuracyDeviceContract(
        model_id=metadata["model_id"],
        variation_percent=metadata["device_variation_percent"],
        device_seed=parser.getint("SNN RouteA", "Device_Seed"),
        enable_r_ratio=parser.getboolean("SNN RouteA", "Enable_R_Ratio"),
        enable_variation=parser.getboolean("SNN RouteA", "Enable_Variation"),
        enable_saf=parser.getboolean("SNN RouteA", "Enable_SAF"),
        saf_percent=(saf[0], saf[1]),
        variation_model=variation_model,
        programming_error_slope=(
            float(programming_error["slope"])
            if "slope" in programming_error
            else None
        ),
        programming_error_intercept_us=(
            float(programming_error["intercept_us"])
            if "intercept_us" in programming_error
            else None
        ),
    )
    if contract.device_seed != FINAL_RRAM_DEVICE_SEED:
        raise FinalRramContractError("formal SNN accuracy Device_Seed must be 112")
    if not contract.enable_r_ratio or not contract.enable_variation:
        raise FinalRramContractError(
            "formal SNN accuracy requires R-ratio and variation"
        )
    if contract.enable_saf or contract.saf_percent != (0.0, 0.0):
        raise FinalRramContractError(
            "formal SNN accuracy requires SAF=0,0 and disabled"
        )
    return contract


def _metadata(model: dict[str, Any], model_reference: str) -> dict[str, Any]:
    device = _mapping(model, "device")
    source = _mapping(model, "source")
    variation = _mapping(model, "variation")
    capabilities = _capabilities(model)
    pulse_width = device.get("pulse_width_ns")
    return {
        "model_id": str(model["model_id"]),
        "model_json": model_reference,
        "source_file": _source_file(source),
        "source_sha256": _source_sha256(source),
        "levels": int(device["levels"]),
        "pulse_width_ns": float(pulse_width) if pulse_width is not None else None,
        "read_voltage_v": float(device["read_voltage_v"]),
        "device_variation_percent": _variation_percent(variation),
        "variation_model": str(
            variation.get("model", "multiplicative_gaussian_resistance")
        ),
        "hardware_write_accounting_supported": _hardware_write_supported(model),
        "iv_nonlinearity_integration": str(
            capabilities.get("iv_nonlinearity", "not_available")
        ),
    }


def _capabilities(model: dict[str, Any]) -> dict[str, Any]:
    item = model.get("capabilities")
    if item is None:
        return {}
    if not isinstance(item, dict):
        raise FinalRramContractError(
            "final RRAM model field 'capabilities' must be an object"
        )
    return item


def _hardware_write_supported(model: dict[str, Any]) -> bool:
    """Require a complete transition primitive, not just a capability flag."""

    capabilities = _capabilities(model)
    if not bool(capabilities.get("hardware_write_accounting", True)):
        return False
    device = _mapping(model, "device")
    levels = _mapping(model, "levels")
    return _has_transition_matrix(model) or (
        device.get("pulse_width_ns") is not None
        and device.get("programming_voltage_abs_v") is not None
        and isinstance(levels.get("pulse"), list)
        and len(levels["pulse"]) == int(device["levels"])
    )


def _has_transition_matrix(model: dict[str, Any]) -> bool:
    transition = model.get("transition_cost")
    if not isinstance(transition, dict):
        return False
    device = _mapping(model, "device")
    try:
        expected = int(device["levels"]) ** 2
    except (KeyError, TypeError, ValueError):
        return False
    return (
        isinstance(transition.get("latency_s"), list)
        and isinstance(transition.get("energy_j"), list)
        and len(transition["latency_s"]) == expected
        and len(transition["energy_j"]) == expected
    )


def _source_file(source: dict[str, Any]) -> str:
    return str(source.get("file") or source.get("programming_table") or "unknown")


def _source_sha256(source: dict[str, Any]) -> str:
    return str(
        source.get("sha256")
        or source.get("levels_sha256")
        or source.get("source_sha256")
        or "unknown"
    )


def _nominal_fit(model: dict[str, Any]) -> dict[str, Any]:
    value = model.get("nominal_fit")
    if isinstance(value, dict):
        return value
    return {"method": "hspice_sequential_program_readback"}


def _reset_model(model: dict[str, Any]) -> dict[str, Any]:
    value = model.get("reset_model")
    if isinstance(value, dict):
        return value
    return {"status": "unavailable_accuracy_only"}


def _variation_percent(variation: dict[str, Any]) -> float:
    return float(variation.get("device_variation_percent", 0.0))


def _variation_estimator(variation: dict[str, Any]) -> str:
    return str(variation.get("estimator", "not_calibrated"))


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise FinalRramContractError(f"final RRAM model field {key!r} must be an object")
    return item


def _list(value: dict[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise FinalRramContractError(f"final RRAM model field {key!r} must be an array")
    return item


def _number(value: float) -> str:
    return f"{float(value):.12g}"


def _actual(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> str:
    try:
        return parser.get(section, option)
    except configparser.Error as exc:
        raise FinalRramContractError(
            f"{source} is missing required final-RRAM field [{section}] {option}"
        ) from exc


def _require_exact(
    actual: configparser.ConfigParser,
    expected: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> None:
    actual_value = _actual(actual, section, option, source).strip()
    expected_value = expected.get(section, option).strip()
    if actual_value != expected_value:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be {expected_value!r}, "
            f"got {actual_value!r}"
        )


def _require_exact_int(
    actual: configparser.ConfigParser,
    expected: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> None:
    try:
        actual_value = int(_actual(actual, section, option, source))
        expected_value = expected.getint(section, option)
    except ValueError as exc:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be an integer"
        ) from exc
    if actual_value != expected_value:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be {expected_value}, got {actual_value}"
        )


def _require_close(
    actual: configparser.ConfigParser,
    expected: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> None:
    try:
        actual_value = float(_actual(actual, section, option, source))
        expected_value = expected.getfloat(section, option)
    except ValueError as exc:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be numeric"
        ) from exc
    if not math.isclose(actual_value, expected_value, rel_tol=1e-11, abs_tol=1e-15):
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be {expected_value:g}, "
            f"got {actual_value:g}"
        )


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _require_float_vector(
    actual: configparser.ConfigParser,
    expected: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> None:
    try:
        actual_values = [float(value) for value in _csv(_actual(actual, section, option, source))]
        expected_values = [float(value) for value in _csv(expected.get(section, option))]
    except ValueError as exc:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be a numeric vector"
        ) from exc
    if len(actual_values) != len(expected_values) or any(
        not math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-15)
        for left, right in zip(actual_values, expected_values)
    ):
        raise FinalRramContractError(
            f"{source} [{section}] {option} does not match the locked model"
        )


def _require_int_vector(
    actual: configparser.ConfigParser,
    expected: configparser.ConfigParser,
    section: str,
    option: str,
    source: str,
) -> None:
    try:
        actual_values = [int(value) for value in _csv(_actual(actual, section, option, source))]
        expected_values = [int(value) for value in _csv(expected.get(section, option))]
    except ValueError as exc:
        raise FinalRramContractError(
            f"{source} [{section}] {option} must be an integer vector"
        ) from exc
    if actual_values != expected_values:
        raise FinalRramContractError(
            f"{source} [{section}] {option} does not match the locked model"
        )


__all__ = [
    "FINAL_RRAM_DEVICE_SEED",
    "FINAL_RRAM_MODEL_ALIASES",
    "FINAL_RRAM_MODEL_IDS",
    "FinalRramContractError",
    "SNNAccuracyDeviceContract",
    "apply_final_rram_model",
    "final_rram_model_path",
    "load_final_rram_model",
    "load_snn_accuracy_device_contract",
    "normalize_final_rram_model_id",
    "require_final_rram_hardware_write_support",
    "validate_final_rram_parser",
    "validate_final_rram_simconfig",
]

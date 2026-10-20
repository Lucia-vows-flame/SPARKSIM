"""Bit-exact functional model and private state store for digital LIF."""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable

from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    CompiledLifConstants,
    DigitalDomainError,
)
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.JointModule import JointModule
from MNSIM.Hardware_Model.Reg import reg
from MNSIM.Hardware_Model.ShiftReg import shiftreg


BETA_Q15 = 24130
BETA_FRACTION_BITS = 15


class DigitalLifError(ValueError):
    """Raised when functional LIF execution violates its integer contract."""


@dataclass(frozen=True)
class InvocationKey:
    sample_id: str
    node_id: str
    timestep: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "node_id": self.node_id,
            "timestep": self.timestep,
        }


@dataclass(frozen=True)
class DigitalLifRequest:
    invocation: InvocationKey
    sequence_length: int
    input_codes: tuple[int, ...]
    previous_state_codes: tuple[int, ...]
    domain: AccumulatorDomain
    constants: CompiledLifConstants
    neuron_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation.to_dict(),
            "sequence_length": self.sequence_length,
            "input_codes": list(self.input_codes),
            "previous_state_codes": list(self.previous_state_codes),
            "domain": self.domain.to_dict(),
            "constants": self.constants.to_dict(),
            "neuron_count": self.neuron_count,
        }


@dataclass(frozen=True)
class DigitalLifResult:
    invocation: InvocationKey
    sequence_length: int
    domain: AccumulatorDomain
    input_codes: tuple[int, ...]
    previous_state_codes: tuple[int, ...]
    delta_codes: tuple[int, ...]
    product_codes: tuple[int, ...]
    scaled_codes: tuple[int, ...]
    unreset_state_codes: tuple[int, ...]
    spike_codes: tuple[int, ...]
    next_state_codes: tuple[int, ...]
    state_buffer_write_required: bool

    @property
    def update_count(self) -> int:
        return len(self.input_codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation.to_dict(),
            "sequence_length": self.sequence_length,
            "domain": self.domain.to_dict(),
            "input_codes": list(self.input_codes),
            "previous_state_codes": list(self.previous_state_codes),
            "delta_codes": list(self.delta_codes),
            "product_codes": list(self.product_codes),
            "scaled_codes": list(self.scaled_codes),
            "unreset_state_codes": list(self.unreset_state_codes),
            "spike_codes": list(self.spike_codes),
            "next_state_codes": list(self.next_state_codes),
            "state_buffer_write_required": self.state_buffer_write_required,
            "update_count": self.update_count,
        }


@dataclass
class LifStateRecord:
    sample_id: str
    lif_node_id: str
    state_shape: tuple[int, ...]
    state_bit_width: int
    domain: AccumulatorDomain
    initial_state_code: int
    state_codes: tuple[int, ...]
    sequence_length: int
    last_completed_timestep: int = -1
    invocation_count: int = 0
    storage_backend: str = "tile_buffer"
    initialization_policy: str = "compiled_reset_per_sample"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "lif_node_id": self.lif_node_id,
            "state_shape": list(self.state_shape),
            "state_bit_width": self.state_bit_width,
            "domain": self.domain.to_dict(),
            "initial_state_code": self.initial_state_code,
            "state_codes": list(self.state_codes),
            "sequence_length": self.sequence_length,
            "last_completed_timestep": self.last_completed_timestep,
            "invocation_count": self.invocation_count,
            "storage_backend": self.storage_backend,
            "initialization_policy": self.initialization_policy,
        }


@dataclass(frozen=True)
class DigitalLifCost:
    bit_width: int
    beta_q15: int
    csd_terms: tuple[tuple[int, int], ...]
    ownership_by_tile: dict[int, int]
    neuron_count: int
    active_tile_count: int
    lanes_per_tile: int
    batches_by_tile: dict[int, int]
    batches_per_step: int
    sequence_length: int
    update_count: int
    state_bytes_by_tile: dict[int, int]
    compute_latency_ns: float
    compute_energy_nj: float
    state_read_latency_ns: float
    state_read_energy_nj: float
    state_write_latency_ns: float
    state_write_energy_nj: float
    provisioned_area_um2: float
    operation_inventory: dict[str, Any]
    slice_counts: dict[str, int]

    @property
    def total_latency_ns(self) -> float:
        return (
            self.compute_latency_ns
            + self.state_read_latency_ns
            + self.state_write_latency_ns
        )

    @property
    def total_energy_nj(self) -> float:
        return (
            self.compute_energy_nj
            + self.state_read_energy_nj
            + self.state_write_energy_nj
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "csd_terms": [list(term) for term in self.csd_terms],
            "total_latency_ns": self.total_latency_ns,
            "total_energy_nj": self.total_energy_nj,
        }


class DigitalLifCostModel:
    CSD_TERMS = ((1, 15), (-1, 13), (-1, 9), (1, 6), (1, 1))

    def __init__(self, sim_config_path: str):
        self.sim_config_path = str(sim_config_path)
        config = configparser.ConfigParser()
        if not config.read(self.sim_config_path):
            raise DigitalLifError(f"cannot read SimConfig {self.sim_config_path}")
        if not config.has_section("Digital LIF"):
            raise DigitalLifError("SimConfig is missing [Digital LIF]")
        section = config["Digital LIF"]
        required = {
            "Digital_LIF_Enable": "1",
            "Digital_LIF_InputDomain": "inherit_completed_accumulator",
            "Digital_LIF_StateDomain": "inherit_completed_accumulator",
            "Digital_LIF_AllowInputRequantization": "0",
            "Digital_LIF_CoefficientBit": "16",
            "Digital_LIF_CoefficientFractionBit": "15",
            "Digital_LIF_DecayMode": "csd_shift_add",
            "Digital_LIF_StateStorage": "tile_buffer",
            "Digital_LIF_ResetMode": "hard",
            "Digital_LIF_RoundingMode": "nearest_even",
            "Digital_LIF_ConstantCompileRoundingMode": "nearest_even",
            "Digital_LIF_OverflowMode": "error",
            "Digital_LIF_InitialStateMode": "reset_code",
            "Digital_LIF_SpikeBit": "1",
            "Digital_LIF_WriteTerminalState": "0",
        }
        for key, expected in required.items():
            actual = section.get(key)
            if actual != expected:
                raise DigitalLifError(
                    f"unsupported [Digital LIF] {key}={actual!r}; expected {expected!r}"
                )
        self.lanes_per_tile = _positive_int(
            section.getint("Digital_LIF_LanesPerTile"),
            "Digital_LIF_LanesPerTile",
        )

    def cost_sequence(
        self,
        *,
        bit_width: int,
        ownership_by_tile: dict[int, int],
        sequence_length: int,
        provisioned_tile_count: int,
        beta_q15: int = BETA_Q15,
    ) -> DigitalLifCost:
        bit_width = _positive_int(bit_width, "bit_width")
        sequence_length = _positive_int(sequence_length, "sequence_length")
        provisioned_tile_count = _positive_int(
            provisioned_tile_count, "provisioned_tile_count"
        )
        beta_q15 = _integer(beta_q15, "beta_q15")
        if beta_q15 != BETA_Q15:
            raise DigitalLifError(f"unsupported beta_q15 {beta_q15}")
        ownership = self._validate_ownership(
            ownership_by_tile, provisioned_tile_count
        )
        neuron_count = sum(ownership.values())
        batches = {
            tile_id: math.ceil(count / self.lanes_per_tile)
            for tile_id, count in ownership.items()
        }
        batches_per_step = max(batches.values())
        state_bytes = {
            tile_id: math.ceil(count * bit_width / 8)
            for tile_id, count in ownership.items()
        }
        primitive = self._lane_primitive_cost(bit_width)
        compute_latency_ns = (
            sequence_length * batches_per_step * primitive["latency_ns"]
        )
        update_count = neuron_count * sequence_length
        compute_energy_nj = update_count * primitive["energy_nj"]
        reads = max(sequence_length - 1, 0)
        writes = max(sequence_length - 1, 0)
        read_latency, read_energy = self._buffer_phase_cost(state_bytes, reads, read=True)
        write_latency, write_energy = self._buffer_phase_cost(
            state_bytes, writes, read=False
        )
        area = (
            provisioned_tile_count
            * self.lanes_per_tile
            * primitive["area_um2"]
        )
        return DigitalLifCost(
            bit_width=bit_width,
            beta_q15=beta_q15,
            csd_terms=self.CSD_TERMS,
            ownership_by_tile=ownership,
            neuron_count=neuron_count,
            active_tile_count=len(ownership),
            lanes_per_tile=self.lanes_per_tile,
            batches_by_tile=batches,
            batches_per_step=batches_per_step,
            sequence_length=sequence_length,
            update_count=update_count,
            state_bytes_by_tile=state_bytes,
            compute_latency_ns=compute_latency_ns,
            compute_energy_nj=compute_energy_nj,
            state_read_latency_ns=read_latency,
            state_read_energy_nj=read_energy,
            state_write_latency_ns=write_latency,
            state_write_energy_nj=write_energy,
            provisioned_area_um2=area,
            operation_inventory=primitive["operation_inventory"],
            slice_counts=primitive["slice_counts"],
        )

    def _validate_ownership(
        self,
        ownership_by_tile: dict[int, int],
        provisioned_tile_count: int,
    ) -> dict[int, int]:
        if not isinstance(ownership_by_tile, dict) or not ownership_by_tile:
            raise DigitalLifError("ownership_by_tile must contain real Tile shards")
        ownership: dict[int, int] = {}
        for tile_id, count in ownership_by_tile.items():
            tile_id = _nonnegative_int(tile_id, "Tile id")
            count = _positive_int(count, f"ownership count for Tile {tile_id}")
            if tile_id >= provisioned_tile_count:
                raise DigitalLifError(
                    f"ownership Tile {tile_id} is outside {provisioned_tile_count} Tiles"
                )
            ownership[tile_id] = count
        return ownership

    def _lane_primitive_cost(self, bit_width: int) -> dict[str, Any]:
        delta_width = bit_width + 1
        product_width = delta_width + 16
        state_width = bit_width + 1
        delta_adder = _adder_cost(self.sim_config_path, delta_width)
        product_adder = _adder_cost(self.sim_config_path, product_width)
        state_adder = _adder_cost(self.sim_config_path, state_width)
        product_shift = _sliced_cost(
            self.sim_config_path, "shiftreg", product_width
        )
        round_proxy = _sliced_cost(
            self.sim_config_path, "jointmodule", product_width
        )
        state_proxy = _sliced_cost(
            self.sim_config_path, "jointmodule", state_width
        )
        state_register = _sliced_cost(self.sim_config_path, "reg", bit_width)
        spike_register = _sliced_cost(self.sim_config_path, "reg", 1)

        energy = (
            delta_adder["energy_nj"]
            + 4 * product_adder["energy_nj"]
            + product_shift["energy_nj"]
            + round_proxy["energy_nj"]
            + delta_adder["energy_nj"]
            + state_adder["energy_nj"]
            + 2 * state_proxy["energy_nj"]
            + 3 * state_register["energy_nj"]
            + spike_register["energy_nj"]
        )
        area = (
            delta_adder["area_um2"]
            + 4 * product_adder["area_um2"]
            + product_shift["area_um2"]
            + round_proxy["area_um2"]
            + delta_adder["area_um2"]
            + state_adder["area_um2"]
            + 2 * state_proxy["area_um2"]
            + 3 * state_register["area_um2"]
            + spike_register["area_um2"]
        )
        latency = (
            state_register["latency_ns"]
            + delta_adder["latency_ns"]
            + 3 * product_adder["latency_ns"]
            + product_shift["latency_ns"]
            + round_proxy["latency_ns"]
            + delta_adder["latency_ns"]
            + state_adder["latency_ns"]
            + state_proxy["latency_ns"]
            + state_proxy["latency_ns"]
            + max(state_register["latency_ns"], spike_register["latency_ns"])
        )
        return {
            "energy_nj": energy,
            "area_um2": area,
            "latency_ns": latency,
            "operation_inventory": {
                "adder_count": 7,
                "shiftreg_count": 1,
                "jointmodule_proxy_count": 3,
                "proxy_types": [
                    "nearest_even_decision",
                    "threshold_comparator",
                    "hard_reset_mux",
                ],
                "register_bits_per_update": 3 * bit_width + 1,
                "delta_width": delta_width,
                "product_width": product_width,
                "state_width": state_width,
            },
            "slice_counts": {
                "coefficient_shiftreg": product_shift["slice_count"],
                "round_decision_jointmodule": round_proxy["slice_count"],
                "state_jointmodule": state_proxy["slice_count"],
                "state_register": state_register["slice_count"],
                "spike_register": spike_register["slice_count"],
            },
        }

    def _buffer_phase_cost(
        self,
        state_bytes_by_tile: dict[int, int],
        repetition_count: int,
        *,
        read: bool,
    ) -> tuple[float, float]:
        if repetition_count == 0:
            return 0.0, 0.0
        latencies: list[float] = []
        energies: list[float] = []
        for byte_count in state_bytes_by_tile.values():
            state_buffer = buffer(
                self.sim_config_path,
                buf_level=2,
                default_buf_size=max(1, math.ceil(byte_count / 1024)),
            )
            if read:
                state_buffer.calculate_buf_read_latency(rdata=byte_count)
                state_buffer.calculate_buf_read_energy(rdata=byte_count)
                latencies.append(state_buffer.buf_rlatency)
                energies.append(state_buffer.buf_renergy)
            else:
                state_buffer.calculate_buf_write_latency(wdata=byte_count)
                state_buffer.calculate_buf_write_energy(wdata=byte_count)
                latencies.append(state_buffer.buf_wlatency)
                energies.append(state_buffer.buf_wenergy)
        return repetition_count * max(latencies), repetition_count * sum(energies)


def round_signed_nearest_even(numerator: int, shift: int) -> int:
    """Round numerator / 2**shift to nearest integer, with ties to even."""
    value = _integer(numerator, "numerator")
    shift = _nonnegative_int(shift, "shift")
    if shift == 0:
        return value
    denominator = 1 << shift
    floor_value = value // denominator
    remainder = value - floor_value * denominator
    half = denominator >> 1
    if remainder > half or (remainder == half and floor_value % 2):
        return floor_value + 1
    return floor_value


def checked_narrow_signed(value: int, bit_width: int, *, name: str) -> int:
    value = _integer(value, name)
    bit_width = _positive_int(bit_width, "bit_width")
    minimum = -(1 << (bit_width - 1))
    maximum = (1 << (bit_width - 1)) - 1
    if not minimum <= value <= maximum:
        raise DigitalLifError(
            f"{name} code {value} is outside signed {bit_width}-bit range"
        )
    return value


def execute_digital_lif_update(request: DigitalLifRequest) -> DigitalLifResult:
    _validate_request(request)
    width = request.domain.signed_bit_width
    inputs = tuple(
        checked_narrow_signed(value, width, name=f"input_codes[{index}]")
        for index, value in enumerate(request.input_codes)
    )
    previous = tuple(
        checked_narrow_signed(value, width, name=f"previous_state_codes[{index}]")
        for index, value in enumerate(request.previous_state_codes)
    )

    deltas: list[int] = []
    products: list[int] = []
    scaled_values: list[int] = []
    unreset_values: list[int] = []
    spikes: list[int] = []
    next_values: list[int] = []
    for index, (input_code, previous_code) in enumerate(zip(inputs, previous)):
        delta = input_code - previous_code
        product = delta * BETA_Q15
        scaled = round_signed_nearest_even(product, BETA_FRACTION_BITS)
        unreset = previous_code + scaled
        spike = int(unreset >= request.constants.threshold_code)
        next_wide = request.constants.reset_code if spike else unreset
        next_code = checked_narrow_signed(
            next_wide,
            width,
            name=f"next_state_codes[{index}]",
        )
        deltas.append(delta)
        products.append(product)
        scaled_values.append(scaled)
        unreset_values.append(unreset)
        spikes.append(spike)
        next_values.append(next_code)

    return DigitalLifResult(
        invocation=request.invocation,
        sequence_length=request.sequence_length,
        domain=request.domain,
        input_codes=inputs,
        previous_state_codes=previous,
        delta_codes=tuple(deltas),
        product_codes=tuple(products),
        scaled_codes=tuple(scaled_values),
        unreset_state_codes=tuple(unreset_values),
        spike_codes=tuple(spikes),
        next_state_codes=tuple(next_values),
        state_buffer_write_required=(
            request.invocation.timestep < request.sequence_length - 1
        ),
    )


class LifStateStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], LifStateRecord] = {}

    def execute(
        self,
        *,
        sample_id: str,
        lif_node_id: str,
        timestep: int,
        sequence_length: int,
        input_codes: Iterable[int],
        state_shape: Iterable[int],
        domain: AccumulatorDomain,
        constants: CompiledLifConstants,
    ) -> DigitalLifResult:
        sample_id = str(sample_id)
        lif_node_id = str(lif_node_id)
        timestep = _nonnegative_int(timestep, "timestep")
        sequence_length = _positive_int(sequence_length, "sequence_length")
        shape = tuple(_positive_int(value, "state_shape dimension") for value in state_shape)
        codes = tuple(_integer(value, "input code") for value in input_codes)
        if _element_count(shape) != len(codes):
            raise DigitalLifError(
                f"state shape {shape} has {_element_count(shape)} elements, "
                f"but received {len(codes)} input codes"
            )
        key = (sample_id, lif_node_id)
        record = self._records.get(key)
        if record is None:
            if timestep != 0:
                raise DigitalLifError(
                    f"{key} expected timestep 0, got {timestep}"
                )
            _require_constants_domain(constants, domain)
            initial = checked_narrow_signed(
                constants.initial_state_code,
                domain.signed_bit_width,
                name="initial_state_code",
            )
            record = LifStateRecord(
                sample_id=sample_id,
                lif_node_id=lif_node_id,
                state_shape=shape,
                state_bit_width=domain.signed_bit_width,
                domain=domain,
                initial_state_code=initial,
                state_codes=(initial,) * len(codes),
                sequence_length=sequence_length,
            )
            self._records[key] = record
        else:
            self._validate_existing_record(
                record=record,
                timestep=timestep,
                sequence_length=sequence_length,
                shape=shape,
                domain=domain,
                constants=constants,
            )

        expected_timestep = record.last_completed_timestep + 1
        if timestep != expected_timestep:
            raise DigitalLifError(
                f"{key} expected timestep {expected_timestep}, got {timestep}"
            )
        result = execute_digital_lif_update(
            DigitalLifRequest(
                invocation=InvocationKey(sample_id, lif_node_id, timestep),
                sequence_length=sequence_length,
                input_codes=codes,
                previous_state_codes=record.state_codes,
                domain=domain,
                constants=constants,
            )
        )
        record.state_codes = result.next_state_codes
        record.last_completed_timestep = timestep
        record.invocation_count += 1
        return result

    def require_complete(
        self,
        sample_id: str,
        lif_node_id: str,
        *,
        sequence_length: int,
    ) -> LifStateRecord:
        key = (str(sample_id), str(lif_node_id))
        if key not in self._records:
            raise DigitalLifError(f"missing LIF state record for {key}")
        record = self._records[key]
        expected = _positive_int(sequence_length, "sequence_length")
        if record.invocation_count != expected:
            raise DigitalLifError(
                f"{key} expected {expected} invocations, got {record.invocation_count}"
            )
        if record.last_completed_timestep != expected - 1:
            raise DigitalLifError(
                f"{key} expected terminal timestep {expected - 1}, got "
                f"{record.last_completed_timestep}"
            )
        return record

    def _validate_existing_record(
        self,
        *,
        record: LifStateRecord,
        timestep: int,
        sequence_length: int,
        shape: tuple[int, ...],
        domain: AccumulatorDomain,
        constants: CompiledLifConstants,
    ) -> None:
        expected_timestep = record.last_completed_timestep + 1
        if timestep != expected_timestep:
            raise DigitalLifError(
                f"{(record.sample_id, record.lif_node_id)} expected timestep "
                f"{expected_timestep}, got {timestep}"
            )
        try:
            record.domain.require_bit_exact_compatible(domain)
        except DigitalDomainError as exc:
            raise DigitalLifError(str(exc)) from exc
        _require_constants_domain(constants, domain)
        if record.sequence_length != sequence_length:
            raise DigitalLifError(
                f"sequence length mismatch: {record.sequence_length} != {sequence_length}"
            )
        if record.state_shape != shape:
            raise DigitalLifError(f"state shape mismatch: {record.state_shape} != {shape}")
        if record.initial_state_code != constants.initial_state_code:
            raise DigitalLifError("initial state code changed during a sequence")


def _validate_request(request: DigitalLifRequest) -> None:
    if not isinstance(request, DigitalLifRequest):
        raise DigitalLifError("request must be a DigitalLifRequest")
    if not isinstance(request.invocation, InvocationKey):
        raise DigitalLifError("invocation must be an InvocationKey")
    sequence_length = _positive_int(request.sequence_length, "sequence_length")
    timestep = _nonnegative_int(request.invocation.timestep, "timestep")
    if timestep >= sequence_length:
        raise DigitalLifError(
            f"timestep {timestep} is outside sequence length {sequence_length}"
        )
    if len(request.input_codes) != len(request.previous_state_codes):
        raise DigitalLifError("input and previous-state code counts differ")
    if request.neuron_count is not None:
        neuron_count = _positive_int(request.neuron_count, "neuron_count")
        if len(request.input_codes) != neuron_count:
            raise DigitalLifError(
                f"request declares {neuron_count} neurons but carries "
                f"{len(request.input_codes)} input codes"
            )
    _require_constants_domain(request.constants, request.domain)


def _require_constants_domain(
    constants: CompiledLifConstants,
    domain: AccumulatorDomain,
) -> None:
    if not isinstance(constants, CompiledLifConstants):
        raise DigitalLifError("constants must be CompiledLifConstants")
    try:
        domain.require_bit_exact_compatible(constants.domain)
    except DigitalDomainError as exc:
        raise DigitalLifError(str(exc)) from exc
    if constants.initial_state_code != constants.reset_code:
        raise DigitalLifError("initial_state_code must equal reset_code")


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _adder_cost(sim_config_path: str, bit_width: int) -> dict[str, float]:
    unit = adder(sim_config_path, bitwidth=bit_width)
    unit.calculate_adder_area()
    unit.calculate_adder_energy()
    return {
        "latency_ns": unit.adder_latency,
        "energy_nj": unit.adder_energy,
        "area_um2": unit.adder_area,
    }


def _sliced_cost(
    sim_config_path: str,
    primitive: str,
    bit_width: int,
) -> dict[str, float | int]:
    remaining = _positive_int(bit_width, f"{primitive} bit_width")
    slices: list[int] = []
    while remaining > 0:
        slice_width = min(remaining, 16)
        slices.append(slice_width)
        remaining -= slice_width
    latency_ns = 0.0
    energy_nj = 0.0
    area_um2 = 0.0
    for slice_width in slices:
        if primitive == "shiftreg":
            unit = shiftreg(sim_config_path, max_shiftbase=slice_width)
            unit.calculate_shiftreg_area()
            unit.calculate_shiftreg_energy()
            latency = unit.shiftreg_latency
            energy = unit.shiftreg_energy
            area = unit.shiftreg_area
        elif primitive == "jointmodule":
            unit = JointModule(sim_config_path, max_bitwidth=slice_width)
            unit.calculate_jointmodule_area()
            unit.calculate_jointmodule_energy()
            latency = unit.jointmodule_latency
            energy = unit.jointmodule_energy
            area = unit.jointmodule_area
        elif primitive == "reg":
            unit = reg(sim_config_path, bitwidth=slice_width)
            unit.calculate_reg_area()
            unit.calculate_reg_energy()
            latency = unit.reg_latency
            energy = unit.reg_energy
            area = unit.reg_area
        else:
            raise DigitalLifError(f"unsupported sliced primitive {primitive}")
        latency_ns = max(latency_ns, latency)
        energy_nj += energy
        area_um2 += area
    return {
        "latency_ns": latency_ns,
        "energy_nj": energy_nj,
        "area_um2": area_um2,
        "slice_count": len(slices),
    }


def _positive_int(value: int, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise DigitalLifError(f"{name} must be positive")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise DigitalLifError(f"{name} must be nonnegative")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DigitalLifError(f"{name} must be an integer")
    return int(value)


__all__ = [
    "BETA_Q15",
    "BETA_FRACTION_BITS",
    "DigitalLifError",
    "InvocationKey",
    "DigitalLifRequest",
    "DigitalLifResult",
    "LifStateRecord",
    "LifStateStore",
    "DigitalLifCost",
    "DigitalLifCostModel",
    "round_signed_nearest_even",
    "checked_narrow_signed",
    "execute_digital_lif_update",
]

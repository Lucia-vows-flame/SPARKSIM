"""Serializable digital-domain contracts for completed accumulators and LIF."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any


class DigitalDomainError(ValueError):
    """Raised when digital-domain metadata is invalid or incompatible."""


@dataclass(frozen=True)
class AccumulatorDomain:
    signed_bit_width: int
    scale_numerator: int
    scale_denominator: int
    domain_id: str
    source: str = "completed_accumulator"

    @classmethod
    def from_scale(
        cls,
        *,
        signed_bit_width: int,
        scale: Fraction | int,
        source: str = "completed_accumulator",
    ) -> "AccumulatorDomain":
        width = _positive_int(signed_bit_width, "signed_bit_width")
        fraction = _positive_fraction(scale, "scale")
        source = str(source)
        payload = {
            "signed_bit_width": width,
            "scale_numerator": fraction.numerator,
            "scale_denominator": fraction.denominator,
            "source": source,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        domain_id = "acc:" + hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16]
        return cls(domain_id=domain_id, **payload)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AccumulatorDomain":
        domain = cls(
            signed_bit_width=_positive_int(
                value["signed_bit_width"], "signed_bit_width"
            ),
            scale_numerator=_integer(value["scale_numerator"], "scale_numerator"),
            scale_denominator=_positive_int(
                value["scale_denominator"], "scale_denominator"
            ),
            domain_id=str(value["domain_id"]),
            source=str(value.get("source", "completed_accumulator")),
        )
        canonical = cls.from_scale(
            signed_bit_width=domain.signed_bit_width,
            scale=domain.scale,
            source=domain.source,
        )
        if canonical.domain_id != domain.domain_id:
            raise DigitalDomainError("domain_id does not match domain metadata")
        return domain

    @property
    def scale(self) -> Fraction:
        return Fraction(self.scale_numerator, self.scale_denominator)

    @property
    def signed_min(self) -> int:
        return -(1 << (self.signed_bit_width - 1))

    @property
    def signed_max(self) -> int:
        return (1 << (self.signed_bit_width - 1)) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "signed_bit_width": self.signed_bit_width,
            "scale_numerator": self.scale_numerator,
            "scale_denominator": self.scale_denominator,
            "domain_id": self.domain_id,
            "source": self.source,
        }

    def require_bit_exact_compatible(self, other: "AccumulatorDomain") -> None:
        if not isinstance(other, AccumulatorDomain) or other != self:
            raise DigitalDomainError(
                f"domain mismatch: {self.to_dict()} != "
                f"{getattr(other, 'to_dict', lambda: other)()}"
            )


@dataclass(frozen=True)
class CompiledLifConstants:
    threshold_code: int
    reset_code: int
    initial_state_code: int
    source_values: dict[str, str]
    compile_errors: dict[str, str]
    domain: AccumulatorDomain
    rounding_mode: str = "nearest_even"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CompiledLifConstants":
        return cls(
            threshold_code=_integer(value["threshold_code"], "threshold_code"),
            reset_code=_integer(value["reset_code"], "reset_code"),
            initial_state_code=_integer(
                value["initial_state_code"], "initial_state_code"
            ),
            source_values={
                str(key): str(item)
                for key, item in dict(value.get("source_values", {})).items()
            },
            compile_errors={
                str(key): str(item)
                for key, item in dict(value.get("compile_errors", {})).items()
            },
            domain=AccumulatorDomain.from_dict(dict(value["domain"])),
            rounding_mode=str(value.get("rounding_mode", "nearest_even")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_code": self.threshold_code,
            "reset_code": self.reset_code,
            "initial_state_code": self.initial_state_code,
            "source_values": dict(self.source_values),
            "compile_errors": dict(self.compile_errors),
            "domain": self.domain.to_dict(),
            "rounding_mode": self.rounding_mode,
        }


def compile_lif_constants(
    *,
    domain: AccumulatorDomain,
    threshold: Fraction | int,
    reset: Fraction | int,
) -> CompiledLifConstants:
    if not isinstance(domain, AccumulatorDomain):
        raise DigitalDomainError("domain must be an AccumulatorDomain")
    threshold_value = _fraction(threshold, "threshold")
    reset_value = _fraction(reset, "reset")
    threshold_code = _compile_code(threshold_value, domain, "threshold")
    reset_code = _compile_code(reset_value, domain, "reset")
    return CompiledLifConstants(
        threshold_code=threshold_code,
        reset_code=reset_code,
        initial_state_code=reset_code,
        source_values={
            "threshold": str(threshold_value),
            "reset": str(reset_value),
            "initial_state": str(reset_value),
        },
        compile_errors={
            "threshold": str(threshold_code * domain.scale - threshold_value),
            "reset": str(reset_code * domain.scale - reset_value),
            "initial_state": str(reset_code * domain.scale - reset_value),
        },
        domain=domain,
    )


def _compile_code(value: Fraction, domain: AccumulatorDomain, name: str) -> int:
    code = _round_fraction_nearest_even(value / domain.scale)
    if not domain.signed_min <= code <= domain.signed_max:
        raise DigitalDomainError(
            f"{name} code {code} is outside signed {domain.signed_bit_width}-bit range"
        )
    return code


def _round_fraction_nearest_even(value: Fraction) -> int:
    floor_value = value.numerator // value.denominator
    remainder = value - floor_value
    half = Fraction(1, 2)
    if remainder > half or (remainder == half and floor_value % 2):
        return floor_value + 1
    return floor_value


def _positive_int(value: int, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise DigitalDomainError(f"{name} must be positive")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DigitalDomainError(f"{name} must be an integer")
    return int(value)


def _fraction(value: Fraction | int, name: str) -> Fraction:
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise DigitalDomainError(f"{name} must be rational") from exc


def _positive_fraction(value: Fraction | int, name: str) -> Fraction:
    result = _fraction(value, name)
    if result <= 0:
        raise DigitalDomainError(f"{name} must be positive")
    return result


__all__ = [
    "AccumulatorDomain",
    "CompiledLifConstants",
    "DigitalDomainError",
    "compile_lif_constants",
]

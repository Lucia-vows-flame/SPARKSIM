from fractions import Fraction

import pytest

from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    DigitalDomainError,
    compile_lif_constants,
)
from MNSIM.Scheduling.round_types import LocalGraphNode, execution_item_from_dict


def _domain(width=18, scale=Fraction(3, 1024)):
    return AccumulatorDomain.from_scale(
        signed_bit_width=width,
        scale=scale,
        source="completed_accumulator",
    )


def test_accumulator_domain_serializes_deterministically():
    domain = _domain()

    assert domain.to_dict() == {
        "signed_bit_width": 18,
        "scale_numerator": 3,
        "scale_denominator": 1024,
        "domain_id": domain.domain_id,
        "source": "completed_accumulator",
    }
    assert AccumulatorDomain.from_dict(domain.to_dict()) == domain
    assert _domain().domain_id == domain.domain_id


def test_domain_compatibility_requires_id_width_and_scale_match():
    domain = _domain()
    domain.require_bit_exact_compatible(_domain())

    with pytest.raises(DigitalDomainError, match="domain mismatch"):
        domain.require_bit_exact_compatible(_domain(width=19))
    with pytest.raises(DigitalDomainError, match="domain mismatch"):
        domain.require_bit_exact_compatible(_domain(scale=Fraction(1, 256)))


def test_lif_constants_compile_nearest_even_and_record_error():
    domain = _domain(width=12, scale=Fraction(1, 8))
    compiled = compile_lif_constants(
        domain=domain,
        threshold=Fraction(5, 16),
        reset=Fraction(0, 1),
    )

    assert compiled.threshold_code == 2  # 2.5 ties to even
    assert compiled.reset_code == 0
    assert compiled.initial_state_code == 0
    assert compiled.rounding_mode == "nearest_even"
    assert compiled.compile_errors["threshold"] == "-1/16"
    assert compiled.domain == domain


def test_constant_compile_handles_negative_half_tie_to_even():
    compiled = compile_lif_constants(
        domain=_domain(width=8, scale=Fraction(1, 4)),
        threshold=Fraction(-5, 8),
        reset=Fraction(-3, 8),
    )

    assert compiled.threshold_code == -2
    assert compiled.reset_code == -2
    assert compiled.initial_state_code == -2


def test_constant_compile_rejects_out_of_range_code():
    with pytest.raises(DigitalDomainError, match="signed 4-bit"):
        compile_lif_constants(
            domain=_domain(width=4, scale=Fraction(1, 1)),
            threshold=Fraction(8, 1),
            reset=Fraction(0, 1),
        )


def test_local_graph_node_round_trips_optional_digital_metadata():
    node = LocalGraphNode(
        layer_id="L001_lif",
        layer_type="lif",
        node_role="hook",
        node_shape={"N": 64},
        digital_metadata={
            "accumulator_domain": _domain().to_dict(),
            "compiled_lif_constants": compile_lif_constants(
                domain=_domain(), threshold=Fraction(1, 2), reset=0
            ).to_dict(),
        },
    )

    assert node.to_dict()["digital_metadata"]["accumulator_domain"][
        "domain_id"
    ] == _domain().domain_id


def test_local_graph_node_default_keeps_existing_ann_constructor_compatible():
    node = LocalGraphNode("Conv1", "conv", "weighted")

    assert node.digital_metadata == {}

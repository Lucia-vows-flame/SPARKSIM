import pytest

from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    compile_lif_constants,
)
from MNSIM.Scheduling.phase2.digital_lif import (
    BETA_Q15,
    DigitalLifError,
    DigitalLifRequest,
    InvocationKey,
    LifStateStore,
    checked_narrow_signed,
    execute_digital_lif_update,
    round_signed_nearest_even,
)


def _domain(width=8):
    return AccumulatorDomain.from_scale(signed_bit_width=width, scale=1)


def _constants(domain, threshold=10, reset=0):
    return compile_lif_constants(
        domain=domain,
        threshold=threshold,
        reset=reset,
    )


@pytest.mark.parametrize(
    ("numerator", "shift", "expected"),
    [
        (7, 2, 2),
        (5, 2, 1),
        (6, 2, 2),
        (10, 2, 2),
        (-7, 2, -2),
        (-5, 2, -1),
        (-6, 2, -2),
        (-10, 2, -2),
    ],
)
def test_signed_nearest_even_handles_non_ties_and_both_tie_parities(
    numerator, shift, expected
):
    assert round_signed_nearest_even(numerator, shift) == expected


def test_integer_lif_update_uses_exact_adc_derived_input_code():
    domain = _domain()
    request = DigitalLifRequest(
        invocation=InvocationKey("s0", "lif0", 0),
        sequence_length=1,
        input_codes=(7, -7, 0),
        previous_state_codes=(0, 0, 4),
        domain=domain,
        constants=_constants(domain),
    )

    result = execute_digital_lif_update(request)

    assert result.delta_codes == (7, -7, -4)
    assert result.product_codes == tuple(delta * BETA_Q15 for delta in (7, -7, -4))
    assert result.scaled_codes == tuple(
        round_signed_nearest_even(product, 15)
        for product in result.product_codes
    )
    assert result.unreset_state_codes == tuple(
        old + scaled for old, scaled in zip((0, 0, 4), result.scaled_codes)
    )
    assert result.next_state_codes == result.unreset_state_codes
    assert result.spike_codes == (0, 0, 0)


def test_threshold_equality_spikes_and_hard_resets():
    domain = _domain()
    result = execute_digital_lif_update(
        DigitalLifRequest(
            invocation=InvocationKey("s0", "lif0", 0),
            sequence_length=2,
            input_codes=(10,),
            previous_state_codes=(10,),
            domain=domain,
            constants=_constants(domain, threshold=10, reset=-3),
        )
    )

    assert result.unreset_state_codes == (10,)
    assert result.spike_codes == (1,)
    assert result.next_state_codes == (-3,)
    assert result.state_buffer_write_required is True


def test_terminal_timestep_omits_state_buffer_write():
    domain = _domain()
    result = execute_digital_lif_update(
        DigitalLifRequest(
            invocation=InvocationKey("s0", "lif0", 2),
            sequence_length=3,
            input_codes=(0,),
            previous_state_codes=(0,),
            domain=domain,
            constants=_constants(domain),
        )
    )

    assert result.state_buffer_write_required is False


def test_checked_narrowing_reports_overflow_instead_of_clipping_or_wrapping():
    with pytest.raises(DigitalLifError, match="outside signed 4-bit range"):
        checked_narrow_signed(8, 4, name="next_state")
    with pytest.raises(DigitalLifError, match="outside signed 4-bit range"):
        checked_narrow_signed(-9, 4, name="next_state")

    assert checked_narrow_signed(-8, 4, name="next_state") == -8
    assert checked_narrow_signed(7, 4, name="next_state") == 7


def test_input_and_previous_state_width_are_checked_without_saturation():
    domain = _domain(width=4)
    with pytest.raises(DigitalLifError, match=r"input_codes\[0\]"):
        execute_digital_lif_update(
            DigitalLifRequest(
                invocation=InvocationKey("s0", "lif0", 0),
                sequence_length=1,
                input_codes=(8,),
                previous_state_codes=(0,),
                domain=domain,
                constants=_constants(domain, threshold=7),
            )
        )


def test_state_store_initializes_from_reset_and_advances_consecutively():
    domain = _domain()
    constants = _constants(domain, reset=-2)
    store = LifStateStore()

    first = store.execute(
        sample_id="s0",
        lif_node_id="lif0",
        timestep=0,
        sequence_length=2,
        input_codes=(4, 6),
        state_shape=(2,),
        domain=domain,
        constants=constants,
    )
    second = store.execute(
        sample_id="s0",
        lif_node_id="lif0",
        timestep=1,
        sequence_length=2,
        input_codes=(4, 6),
        state_shape=(2,),
        domain=domain,
        constants=constants,
    )

    assert first.previous_state_codes == (-2, -2)
    assert second.previous_state_codes == first.next_state_codes
    record = store.require_complete("s0", "lif0", sequence_length=2)
    assert record.invocation_count == 2
    assert record.last_completed_timestep == 1
    assert record.domain == domain


def test_state_store_rejects_duplicate_skip_and_domain_change():
    domain = _domain()
    store = LifStateStore()
    kwargs = dict(
        sample_id="s0",
        lif_node_id="lif0",
        sequence_length=3,
        input_codes=(0,),
        state_shape=(1,),
        domain=domain,
        constants=_constants(domain),
    )
    store.execute(timestep=0, **kwargs)

    with pytest.raises(DigitalLifError, match="expected timestep 1"):
        store.execute(timestep=0, **kwargs)
    with pytest.raises(DigitalLifError, match="expected timestep 1"):
        store.execute(timestep=2, **kwargs)

    other = _domain(width=9)
    with pytest.raises(DigitalLifError, match="domain mismatch"):
        store.execute(
            timestep=1,
            **{**kwargs, "domain": other, "constants": _constants(other)},
        )


def test_state_store_separates_samples_and_audits_exactly_t_invocations():
    domain = _domain()
    constants = _constants(domain)
    store = LifStateStore()
    for sample_id in ("s0", "s1"):
        store.execute(
            sample_id=sample_id,
            lif_node_id="lif0",
            timestep=0,
            sequence_length=1,
            input_codes=(0,),
            state_shape=(1,),
            domain=domain,
            constants=constants,
        )

    assert store.require_complete("s0", "lif0", sequence_length=1).invocation_count == 1
    assert store.require_complete("s1", "lif0", sequence_length=1).invocation_count == 1
    with pytest.raises(DigitalLifError, match="expected 2 invocations"):
        store.require_complete("s0", "lif0", sequence_length=2)

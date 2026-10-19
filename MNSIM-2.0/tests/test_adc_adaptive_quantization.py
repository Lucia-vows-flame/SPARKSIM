import pytest
import torch

from MNSIM.Interface.quantize import _adc_quantize


def test_readc_profile_is_documented_as_five_bit_only():
    from MNSIM.Accuracy_Model.readc import READC_PROVENANCE
    assert READC_PROVENANCE["model"] == "READC_LLOYD_MAX"
    assert READC_PROVENANCE["doi"] == "10.1038/s41467-025-65233-w"


def test_uniform_mode_remains_legacy_round_and_clip():
    values = torch.tensor([-20.2, -1.6, 0.4, 14.8, 20.0])
    result = _adc_quantize(values, -2, 2, {"ADC_quantization_mode": "UNIFORM"})
    assert torch.equal(result, torch.tensor([-2.0, -2.0, 0.0, 2.0, 2.0]))


def test_adaptive_mode_uses_configured_nonuniform_boundaries():
    values = torch.tensor([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])
    result = _adc_quantize(
        values,
        -3,
        3,
        {
            "ADC_quantization_mode": "ADAPTIVE",
            "ADC_adaptive_thresholds": [-2.0, -0.5, 0.0, 0.5, 2.0, 10.0],
        },
    )
    assert torch.equal(result, torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]))


@pytest.mark.parametrize(
    "thresholds",
    [None, [0.0], [0.0, 0.0, 1.0, 2.0, 3.0]],
)
def test_adaptive_mode_rejects_invalid_threshold_table(thresholds):
    with pytest.raises(ValueError, match="ADC_adaptive_thresholds"):
        _adc_quantize(
            torch.tensor([0.0]),
            -3,
            3,
            {
                "ADC_quantization_mode": "ADAPTIVE",
                "ADC_adaptive_thresholds": thresholds,
            },
        )

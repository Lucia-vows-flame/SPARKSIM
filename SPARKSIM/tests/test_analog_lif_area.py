from pathlib import Path

import pytest

from MNSIM.Hardware_Model.analog_lif_area import estimate_analog_lif_area


ROOT = Path(__file__).resolve().parents[1] / "archive/legacy_master_20260925"
FRONTEND = ROOT / "hspice_win_reference_ccii_direct_vcm_lif_gain4_20260725/01_src/reference_ccii_current_diff_core.sp"
LIF = ROOT / "hspice_win_lif_context_threshold_calibration_20260726/01_src"


def test_netlist_area_has_device_provenance_and_scales_by_neuron():
    estimate = estimate_analog_lif_area(
        frontend_netlist=FRONTEND,
        lif_netlists=[LIF / "comp_ct_simple.sp", LIF / "spike_sr_latch.sp", LIF / "lif_current_rate_cell.sp"],
    )
    assert estimate.transistor_count == 36
    assert estimate.active_device_area_um2 > 0
    assert estimate.area_um2_per_neuron == pytest.approx(estimate.active_device_area_um2 * 2.0)
    assert estimate.for_neurons(3) == pytest.approx(3 * estimate.area_um2_per_neuron)
    assert "reference_ccii_current_diff_core.sp" in estimate.provenance[0]


def test_layout_factor_is_explicit():
    with pytest.raises(ValueError, match="layout_overhead_factor"):
        estimate_analog_lif_area(
            frontend_netlist=FRONTEND,
            lif_netlists=[LIF / "comp_ct_simple.sp"],
            layout_overhead_factor=0.5,
        )

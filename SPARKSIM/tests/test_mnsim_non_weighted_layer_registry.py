from MNSIM.Scheduling.non_weighted_layers import (
    REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES,
    REGISTERED_PHASE1_HOOK_LAYER_TYPES,
    get_non_weighted_layer_spec,
)
from MNSIM.Interface.quantize import StraightLayer, StraightLayerStr
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_structure
from MNSIM.Scheduling.round_types import HOOK_LAYER_TYPES


def _hardware_config():
    return {
        "xbar_size": 128,
        "input_bit": 9,
        "weight_bit": 8,
        "ADC_quantize_bit": 10,
        "DAC_num": 64,
        "xbar_polarity": 2,
    }


def _quantize_config():
    return {
        "activation_bit": 9,
        "weight_bit": 8,
    }


def test_lif_is_registered_as_mnsim_non_weighted_hook_layer():
    spec = get_non_weighted_layer_spec("lif")

    assert spec is not None
    assert spec.layer_type == "lif"
    assert spec.phase1_node_role == "hook"
    assert "lif" in REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES
    assert "lif" in REGISTERED_PHASE1_HOOK_LAYER_TYPES
    assert "lif" in HOOK_LAYER_TYPES
    assert "lif" in StraightLayerStr


def test_lif_registration_drives_quantize_layer_and_phase1_frontend():
    lif_config = {"type": "lif", "tau": 3.0, "v_th": 0.7, "v_reset": -0.1, "alpha": 1.5}
    straight_layer = StraightLayer(_hardware_config(), lif_config, _quantize_config())

    assert straight_layer.layer.__class__.__name__ == "LIFNode"
    assert straight_layer.layer.tau == 3.0
    assert straight_layer.layer.v_th == 0.7
    assert straight_layer.layer.v_reset == -0.1
    assert straight_layer.layer.alpha == 1.5

    model_ir = build_model_ir_from_mnsim_structure(
        [{"type": "lif", "tau": 3.0, "v_th": 0.7, "v_reset": -0.1, "Inputindex": [-1]}],
        model_id="registered_lif",
        backend_kind="snn",
    )

    layer = model_ir.layers[0]
    assert layer.layer_id == "L000_lif"
    assert layer.layer_type == "lif"
    assert layer.is_weighted is False
    assert layer.weight_ref_id is None
    assert layer.backend_attrs["tau"] == 3.0
    assert layer.backend_attrs["v_th"] == 0.7
    assert layer.backend_attrs["v_reset"] == -0.1

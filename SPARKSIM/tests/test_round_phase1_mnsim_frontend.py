from MNSIM.Scheduling.phase1 import mnsim_frontend
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_structure


def test_mnsim_frontend_exposes_formal_phase1_identity():
    assert getattr(mnsim_frontend, "FRONTEND_PHASE", None) == "phase1"
    assert getattr(mnsim_frontend, "FRONTEND_KIND", None) == "mnsim_networkgraph"


def test_mnsim_frontend_accepts_concat_as_structural_non_weighted_layer():
    model_ir = build_model_ir_from_mnsim_structure(
        [
            {"type": "conv", "Inputindex": [-1], "Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 3},
            {"type": "conv", "Inputindex": [-2], "Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 1},
            {"type": "concat", "Inputindex": [-1, -2], "Outputindex": [1]},
        ],
        model_id="concat_model",
    )

    concat = model_ir.layers[2]
    assert concat.layer_id == "L002_concat"
    assert concat.layer_type == "concat"
    assert concat.is_weighted is False
    assert concat.input_layer_ids == ["L001_conv", "L000_conv"]
    assert concat.weight_ref_id is None


def test_mnsim_frontend_preserves_weighted_layer_fields_needed_by_phase2():
    model_ir = build_model_ir_from_mnsim_structure(
        [
            {
                "type": "conv",
                "Inputindex": [-1],
                "Outputindex": [1],
                "Inputchannel": 3,
                "Inputsize": [32, 32],
                "Kernelsize": 3,
                "Stride": 1,
                "Padding": 1,
                "Depthwise": "normal",
                "Outputchannel": 8,
                "Outputsize": [32, 32],
                "Inputbit": 9,
                "Weightbit": 9,
                "outputbit": 9,
                "row_split_num": 1,
                "weight_bit_split_part": 1,
            },
            {
                "type": "fc",
                "Inputindex": [-1],
                "Infeature": 128,
                "Outfeature": 10,
                "Inputbit": 9,
                "Weightbit": 9,
                "outputbit": 9,
                "row_split_num": 1,
                "weight_bit_split_part": 1,
            },
        ],
        model_id="weighted_fields_model",
    )

    conv = model_ir.layers[0]
    fc = model_ir.layers[1]

    assert conv.shape["Inputsize"] == [32, 32]
    assert conv.shape["Outputsize"] == [32, 32]
    assert conv.shape["Stride"] == 1
    assert conv.shape["Padding"] == 1
    assert conv.shape["Inputbit"] == 9
    assert conv.shape["Weightbit"] == 9
    assert conv.shape["outputbit"] == 9
    assert conv.shape["Depthwise"] == "normal"
    assert fc.shape["Inputbit"] == 9
    assert fc.shape["Weightbit"] == 9
    assert fc.shape["outputbit"] == 9

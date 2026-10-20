from fractions import Fraction

import pytest

from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    compile_lif_constants,
)
from MNSIM.Scheduling.phase1.mnsim_frontend import (
    build_model_ir_from_mnsim_structure,
)
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, SequenceSpec


HW = {
    "Xbar_Size": [128, 128],
    "Tile_Num": [1, 1],
    "PE_Num": [1, 1],
    "Group_Num": 8,
    "Device_Level": 16,
    "Weight_precision": 4,
    "Xbar_Polarity": 2,
}


def _domain(width=18, scale=Fraction(1, 256)):
    return AccumulatorDomain.from_scale(
        signed_bit_width=width,
        scale=scale,
        source="completed_accumulator",
    )


def _layer(
    layer_id,
    layer_type,
    index,
    inputs,
    *,
    domain=None,
    is_weighted=None,
):
    if is_weighted is None:
        is_weighted = layer_type in {"conv", "fc"}
    attrs = {}
    if domain is not None:
        attrs["accumulator_domain"] = domain.to_dict()
    if layer_type == "lif":
        attrs.update(
            {
                "tau": 1.3579523543,
                "v_th": 0.5,
                "v_reset": 0.0,
                "reset_mode": "hard",
                "compiled_lif_constants": compile_lif_constants(
                    domain=domain,
                    threshold=Fraction(1, 2),
                    reset=0,
                ).to_dict(),
            }
        )
    shape = (
        {"Inputchannel": 1, "Outputchannel": 8, "Kernelsize": 1}
        if layer_type == "conv"
        else {"N": 8, "output_neuron_region": [0, 8]}
    )
    return LayerIR(
        layer_id=layer_id,
        layer_type=layer_type,
        is_weighted=is_weighted,
        stable_topological_index=index,
        input_tensor_ids=[f"{item}:out" for item in inputs] or ["input:0"],
        output_tensor_ids=[f"{layer_id}:out"],
        input_layer_ids=list(inputs),
        output_consumer_ids=[],
        weight_ref_id=f"w:{layer_id}" if is_weighted else None,
        shape=shape,
        backend_attrs=attrs,
    )


def _model(layers, recurrent_ids=("L1",), t=4):
    by_id = {layer.layer_id: layer for layer in layers}
    for layer in layers:
        for producer in layer.input_layer_ids:
            by_id[producer].output_consumer_ids.append(layer.layer_id)
    return ModelIR(
        model_id="snn",
        backend_kind="snn",
        sequence_spec=SequenceSpec(t, True, True, list(recurrent_ids)),
        layers=layers,
    )


def test_valid_weighted_to_lif_graph_preserves_digital_metadata():
    domain = _domain()
    model = _model(
        [
            _layer("L0", "conv", 0, [], domain=domain),
            _layer("L1", "lif", 1, ["L0"], domain=domain),
        ]
    )

    plan = build_round_plan(model, HW)

    assert plan.feasible is True
    lif_node = next(
        node
        for item in plan.execution_items
        for node in item.local_graph_nodes
        if node.layer_id == "L1"
    )
    assert lif_node.digital_metadata["accumulator_domain"]["domain_id"] == domain.domain_id
    assert lif_node.digital_metadata["compiled_lif_constants"]["threshold_code"] == 128
    assert lif_node.node_shape["output_neuron_region"] == [0, 8]
    assert all(entry.layer_id != "L1" for item in plan.execution_items for entry in item.selected_entries)


@pytest.mark.parametrize(
    "recurrent_ids,expected",
    [
        (["missing"], "unknown recurrent_state_id"),
        (["L1", "L1"], "duplicate recurrent_state_id"),
        (["L0"], "must resolve to lif"),
    ],
)
def test_invalid_recurrent_state_ids_are_rejected(recurrent_ids, expected):
    domain = _domain()
    model = _model(
        [
            _layer("L0", "conv", 0, [], domain=domain),
            _layer("L1", "lif", 1, ["L0"], domain=domain),
        ],
        recurrent_ids=recurrent_ids,
    )

    plan = build_round_plan(model, HW)

    assert plan.feasible is False
    assert expected in plan.invalid_reason


def test_lif_requires_exactly_one_direct_input():
    domain = _domain()
    for inputs in ([], ["L0", "L2"]):
        layers = [
            _layer("L0", "conv", 0, [], domain=domain),
            _layer("L2", "conv", 1, [], domain=domain),
            _layer("L1", "lif", 2, inputs, domain=domain),
        ]
        plan = build_round_plan(_model(layers), HW)
        assert plan.feasible is False
        assert "exactly one direct input" in plan.invalid_reason


def test_lif_rejects_missing_or_mismatched_input_domain():
    lif_domain = _domain()
    for producer_domain in (None, _domain(width=19)):
        model = _model(
            [
                _layer("L0", "conv", 0, [], domain=producer_domain),
                _layer("L1", "lif", 1, ["L0"], domain=lif_domain),
            ]
        )
        plan = build_round_plan(model, HW)
        assert plan.feasible is False
        assert "accumulator domain" in plan.invalid_reason


def test_same_domain_merge_can_publish_one_domain_to_lif():
    domain = _domain()
    model = _model(
        [
            _layer("L0", "conv", 0, [], domain=domain),
            _layer("L2", "conv", 1, [], domain=domain),
            _layer("ADD", "element_sum", 2, ["L0", "L2"], domain=domain),
            _layer("L1", "lif", 3, ["ADD"], domain=domain),
        ]
    )

    assert build_round_plan(model, HW).feasible is True


def test_frontend_preserves_formal_lif_metadata():
    domain = _domain()
    constants = compile_lif_constants(
        domain=domain, threshold=Fraction(1, 2), reset=0
    )
    model = build_model_ir_from_mnsim_structure(
        [
            {
                "type": "fc",
                "Inputindex": [-1],
                "Infeature": 8,
                "Outfeature": 8,
                "accumulator_domain": domain.to_dict(),
            },
            {
                "type": "lif",
                "Inputindex": [-1],
                "N": 8,
                "output_neuron_region": [0, 8],
                "tau": 1.3579523543,
                "v_th": 0.5,
                "v_reset": 0.0,
                "reset_mode": "hard",
                "accumulator_domain": domain.to_dict(),
                "compiled_lif_constants": constants.to_dict(),
            },
        ],
        model_id="frontend_snn",
        backend_kind="snn",
        sequence_spec=SequenceSpec(4, True, True, ["L001_lif"]),
    )

    lif = model.layers[1]
    assert lif.backend_attrs["accumulator_domain"]["domain_id"] == domain.domain_id
    assert lif.backend_attrs["compiled_lif_constants"]["initial_state_code"] == 0
    assert lif.shape["output_neuron_region"] == [0, 8]

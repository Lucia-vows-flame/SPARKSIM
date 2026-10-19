from pathlib import Path

from MNSIM.Interface.mnsim_non_weighted_layers import (
    get_non_weighted_layer_spec as legacy_get_non_weighted_layer_spec,
)
from MNSIM.Interface.round_phase1_planner import build_round_plan as legacy_build_round_plan
from MNSIM.Interface.round_phase1_types import RoundPlan as LegacyRoundPlan
from MNSIM.Scheduling.non_weighted_layers import get_non_weighted_layer_spec
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, RoundPlan, SequenceSpec


def _model():
    return ModelIR(
        model_id="arch_smoke",
        backend_kind="ann",
        sequence_spec=SequenceSpec(
            sequence_length=1,
            ordered_steps=False,
            cross_step_dependency=False,
            recurrent_state_ids=[],
        ),
        layers=[
            LayerIR(
                layer_id="FC0",
                layer_type="fc",
                is_weighted=True,
                stable_topological_index=0,
                input_tensor_ids=["input:0"],
                output_tensor_ids=["FC0:out"],
                input_layer_ids=[],
                output_consumer_ids=[],
                weight_ref_id="w:FC0",
                shape={"Infeature": 128, "Outfeature": 32},
                backend_attrs={},
            )
        ],
    )


def _config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [1, 1],
        "PE_Num": [1, 1],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 2,
        "Xbar_Polarity": 1,
    }


def test_round_construction_core_lives_under_scheduling_package():
    plan = build_round_plan(_model(), _config())

    assert plan.feasible is True
    assert isinstance(plan, RoundPlan)
    assert RoundPlan.__module__ == "MNSIM.Scheduling.round_types"
    assert build_round_plan.__module__ == "MNSIM.Scheduling.phase1.round_planner"


def test_interface_phase1_modules_are_compatibility_shims():
    assert legacy_build_round_plan is build_round_plan
    assert LegacyRoundPlan is RoundPlan


def test_non_weighted_layer_registry_lives_under_scheduling_package():
    spec = get_non_weighted_layer_spec("lif")

    assert spec is not None
    assert legacy_get_non_weighted_layer_spec is get_non_weighted_layer_spec
    assert get_non_weighted_layer_spec.__module__ == "MNSIM.Scheduling.non_weighted_layers"


def test_phase2_core_lives_under_scheduling_package():
    import MNSIM.Scheduling.phase2 as phase2

    assert phase2.BoundaryStateManager.__module__ == "MNSIM.Scheduling.phase2.boundary"
    assert phase2.RoundLowerer.__module__ == "MNSIM.Scheduling.phase2.lowerer"
    assert phase2.ReportAggregator.__module__ == "MNSIM.Scheduling.phase2.report"


def test_scheduling_package_does_not_import_interface_package():
    package_root = Path(__file__).resolve().parents[1] / "MNSIM" / "Scheduling"
    offenders = []
    for path in sorted(package_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "MNSIM.Interface" in text:
            offenders.append(str(path.relative_to(package_root)))

    assert offenders == []

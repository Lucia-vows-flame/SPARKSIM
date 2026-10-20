from pathlib import Path
import configparser

import pytest
import torch

from test_v7_residual_snn_network import block_graph
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.round_types import Phase1Invalid

CONFIG = str(Path(__file__).resolve().parents[1] / 'SimConfig.ini')


def calibrated_block(projection):
    torch.manual_seed(7)
    net = block_graph(projection)
    net.train()
    with torch.no_grad():
        net.sequence_qat_forward(torch.randint(0, 2, (2, 3, 2, 8, 8)).float())
    net.eval()
    return net


@pytest.mark.parametrize('projection', [False, True])
def test_residual_real_phase1_phase2(projection):
    net = calibrated_block(projection)
    ir = build_model_ir_from_mnsim_network(net, 'residual', backend_kind='snn', sim_config=CONFIG)
    merge = next(l for l in ir.layers if l.layer_type == 'element_sum')
    final_lif = ir.layers[-1]
    assert final_lif.backend_attrs['accumulator_domain'] == merge.backend_attrs['accumulator_domain']
    assert final_lif.backend_attrs['lif_owner_layer_id'] == 'L003_conv'
    assert all(l.shape['Inputbit'] == 1 for l in ir.layers if l.is_weighted)
    plan = build_round_plan(ir, CONFIG)
    assert plan.feasible, plan.invalid_reason
    report = execute_round_plan(plan, sim_config_path=CONFIG, ops_total=1,
                               ops_count_policy='test', model_inputs={'model_input': {'shape': {'C': 2}}})
    assert report.feasible
    lif_steps = [s for s in report.step_results if s['source_kind'] == 'digital_lif']
    assert len(lif_steps) == 60
    assert report.inference_metrics['energy_j'] > 0
    assert report.inference_metrics['latency_s'] > 0
    assert report.inference_metrics['lif_update_count'] == 30 * 2 * (64 * 2 if not projection else 16 * 4)


def test_residual_requires_calibrated_bn_and_merge_domains():
    with pytest.raises(Phase1Invalid, match='trained/calibrated'):
        build_model_ir_from_mnsim_network(block_graph(), 'residual', backend_kind='snn', sim_config=CONFIG)


@pytest.mark.parametrize('projection', [False, True])
@pytest.mark.parametrize('xbar_size', [128, 16])
def test_residual_cross_round_execution(tmp_path, monkeypatch, projection, xbar_size):
    from test_v7_residual_snn_network import HW
    monkeypatch.setitem(HW, 'xbar_size', xbar_size)
    parser = configparser.ConfigParser()
    parser.read(CONFIG)
    parser.set('Crossbar level', 'Xbar_Size', f'{xbar_size},{xbar_size}')
    parser.set('Crossbar level', 'Subarray_Size', str(xbar_size))
    parser.set('Architecture level', 'Tile_Num', '1,1')
    parser.set('Tile level', 'PE_Num', '1,1')
    parser.set('Process element level', 'Group_Num', '1')
    config = tmp_path / 'small.ini'
    with config.open('w') as handle:
        parser.write(handle)
    net = calibrated_block(projection)
    net.sequence_length = 3
    ir = build_model_ir_from_mnsim_network(net, 'small_residual', backend_kind='snn', sim_config=config)
    plan = build_round_plan(ir, config)
    assert plan.feasible, plan.invalid_reason
    assert len(plan.execution_items) > 1
    report = execute_round_plan(plan, sim_config_path=str(config), ops_total=1,
                               ops_count_policy='test', model_inputs={'model_input': {'shape': {'C': 2}}})
    assert report.feasible
    assert len([s for s in report.step_results if s['source_kind'] == 'digital_lif']) == 6


def test_wrong_lif_owner_is_rejected():
    net = calibrated_block(True)
    ir = build_model_ir_from_mnsim_network(net, 'residual', backend_kind='snn', sim_config=CONFIG)
    ir.layers[-1].backend_attrs['lif_owner_layer_id'] = 'L005_conv'
    plan = build_round_plan(ir, CONFIG)
    assert not plan.feasible
    assert 'invalid weighted owner' in plan.invalid_reason

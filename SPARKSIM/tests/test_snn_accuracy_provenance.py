from types import SimpleNamespace

import MNSIM.Interface.round_orchestrated_runner as runner_module
from MNSIM.Interface.round_orchestrated_runner import RunnerDependencies
from MNSIM.Interface.round_orchestrated_types import RoundOrchestratedRunRequest


class _Context:
    def __init__(self):
        self.hardware_config = {"ADC_quantization_mode": "UNIFORM"}
        self.snn_accuracy_config = SimpleNamespace(
            hardware_profile="frontend_lut_lif_lut",
            lif_perturbation_mode="none",
            lif_perturbation_seed=112,
        )
        self.net = SimpleNamespace(
            layer_list=[
                SimpleNamespace(
                    layer_config={
                        "type": "lif",
                        "tau": 50.5016666556,
                        "v_th": 0.01,
                        "v_reset": 0.0,
                    }
                )
            ]
        )

    def get_net_bits(self):
        return []

    def origin_evaluate(self, **_kwargs):
        return 0.25

    def set_net_bits_evaluate(self, **_kwargs):
        return 0.2

    def test_split_identity(self):
        return {
            "split": "test",
            "resolved_split": "test",
            "sample_count": 4000,
        }


def test_snn_accuracy_report_records_pretrain_and_test_split_provenance(tmp_path):
    context = _Context()
    checkpoint = tmp_path / "standard_vgg8_snn_v7_software_pretrain_best.pth"
    checkpoint.write_bytes(b"pretrain-checkpoint")
    sim_config = tmp_path / "SimConfig.ini"
    sim_config.write_text("[SNN Accuracy]\n", encoding="utf-8")
    request = RoundOrchestratedRunRequest(
        network="standard_vgg8_snn_v7_phase6",
        weights=checkpoint,
        num_classes=8,
        sim_config=sim_config,
        output_dir=tmp_path,
        dataset_module="MNSIM.Interface.dvs_binned_npz",
        run_original=True,
        run_pim=False,
    )
    report = runner_module._evaluate_accuracy(
        request,
        context,
        RunnerDependencies(),
    )

    assert report["snn_accuracy_data"] == {
        "source_checkpoint": str(request.weights),
        "source_checkpoint_kind": "software_pretrain",
        "calibration_split": None,
        "test_split": {
            "split": "test",
            "resolved_split": "test",
            "sample_count": 4000,
        },
    }

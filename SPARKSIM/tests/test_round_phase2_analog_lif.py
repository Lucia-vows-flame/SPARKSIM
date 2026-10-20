import json

import numpy as np
import pytest

from MNSIM.Hardware_Model.analog_lif_lut import load_analog_lif_lut
from MNSIM.Scheduling.phase2.analog_lif import AnalogLifBackend, AnalogLifError
from test_local_analog_lif_lut import _write_artifact


def _load_synthetic_bundle(tmp_path, mutator=None):
    root = _write_artifact(tmp_path / "lut")
    if mutator is not None:
        mutator(root)
    return load_analog_lif_lut(root, verify_checksums=mutator is None)


def test_queries_positive_and_negative_current_rails_and_preserves_signed_lut_energy(
    tmp_path,
):
    def make_energy_signed(root):
        energy_path = root / "lif_current_step20ns" / "energy_rails_j.npy"
        np.save(energy_path, np.array([[5.0, -6.0], [7.0, -8.0]]))

    backend = AnalogLifBackend(_load_synthetic_bundle(tmp_path, make_energy_signed))

    result = backend.execute(
        ipos_a=np.array([1.0, 0.0]),
        ineg_a=np.array([0.0, 2.0]),
        vmem0_v=np.array([0.0, 1.0]),
    )

    assert np.allclose(result.frontend_iout_to_lif_a, [1.0, 2.0])
    assert np.allclose(result.raw_vmem_next_v, [0.2, 0.4])
    assert np.allclose(result.vmem_next, [0.2, 0.4])
    assert result.spike_codes == (1, 0)
    assert result.frontend_latency_s == pytest.approx(2.0)
    assert result.frontend_energy_j == pytest.approx(35.0)
    assert result.lif_latency_s == pytest.approx(2e-8)
    assert result.lif_energy_j == pytest.approx(-2.0)
    assert result.cost.energy_j == pytest.approx(33.0)
    assert result.cost.latency_s == pytest.approx(2.00000002)
    assert result.cost.power_w == pytest.approx(33.0 / 2.00000002)
    assert result.audit_metadata["negative_vmem_policy"] == "error"
    assert result.audit_metadata["query_counts"] == {"frontend": 2, "lif": 2}
    assert result.to_dict()["raw_vmem_next_v"] == pytest.approx([0.2, 0.4])
    assert result.to_dict()["cost"]["energy_j"] == pytest.approx(33.0)


def test_progresses_lut_measured_state_across_two_timesteps(tmp_path):
    backend = AnalogLifBackend(_load_synthetic_bundle(tmp_path))

    first = backend.execute(ipos_a=[1.0], ineg_a=[0.0], vmem0_v=[0.0])
    second = backend.execute(ipos_a=[1.0], ineg_a=[0.0], vmem0_v=first.vmem_next)

    assert first.vmem_next.tolist() == pytest.approx([0.2])
    assert second.raw_vmem_next_v.tolist() == pytest.approx([0.22])
    assert second.vmem_next.tolist() == pytest.approx([0.22])
    assert second.spike_codes == (0,)


def test_clamp_zero_policy_reports_raw_negative_successor_without_clipping_lut_energy(
    tmp_path,
):
    def make_negative_successor(root):
        np.save(
            root / "lif_current_step20ns" / "vmem_next_v.npy",
            np.array([[-0.5, 0.2], [0.3, 0.4]]),
        )
        np.save(
            root / "lif_current_step20ns" / "energy_rails_j.npy",
            np.array([[-5.0, 6.0], [7.0, 8.0]]),
        )

    backend = AnalogLifBackend(
        _load_synthetic_bundle(tmp_path, make_negative_successor),
        negative_vmem_policy="clamp_zero",
    )

    result = backend.execute(ipos_a=[0.0], ineg_a=[0.0], vmem0_v=[0.0])

    assert result.raw_vmem_next_v.tolist() == pytest.approx([-0.5])
    assert result.vmem_next.tolist() == pytest.approx([0.0])
    assert result.lif_energy_j == pytest.approx(-5.0)
    assert result.cost.energy_j == pytest.approx(5.0)
    assert result.audit_metadata["negative_vmem_policy"] == "clamp_zero"
    assert result.audit_metadata["negative_vmem_clamp_count"] == 1
    assert result.audit_metadata["raw_min_clamped_vmem_next_v"] == pytest.approx(-0.5)
    serialized = result.to_dict()
    assert serialized["raw_vmem_next_v"] == pytest.approx([-0.5])
    assert serialized["vmem_next"] == pytest.approx([0.0])


def test_error_policy_rejects_negative_successor_and_shape_mismatches(tmp_path):
    def make_negative_successor(root):
        np.save(
            root / "lif_current_step20ns" / "vmem_next_v.npy",
            np.array([[-0.5, 0.2], [0.3, 0.4]]),
        )

    backend = AnalogLifBackend(_load_synthetic_bundle(tmp_path, make_negative_successor))

    with pytest.raises(AnalogLifError, match="negative Vmem successor"):
        backend.execute(ipos_a=[0.0], ineg_a=[0.0], vmem0_v=[0.0])

    with pytest.raises(AnalogLifError, match="same shape"):
        backend.execute(ipos_a=[0.0, 1.0], ineg_a=[0.0], vmem0_v=[0.0])


def test_cost_uses_lut_latency_and_energy_not_analytic_membrane_integration(tmp_path):
    backend = AnalogLifBackend(_load_synthetic_bundle(tmp_path))

    result = backend.execute(ipos_a=[0.0], ineg_a=[0.0], vmem0_v=[0.0])

    assert result.frontend_latency_s == pytest.approx(1.0)
    assert result.frontend_energy_j == pytest.approx(10.0)
    assert result.lif_latency_s == pytest.approx(2e-8)
    assert result.lif_energy_j == pytest.approx(5.0)
    assert result.cost.latency_s == pytest.approx(1.00000002)
    assert result.cost.energy_j == pytest.approx(15.0)
    assert result.cost.power_w == pytest.approx(15.0 / 1.00000002)
    assert json.loads(json.dumps(result.to_dict()))["audit_metadata"][
        "cost_source"
    ] == "lut_only"


def test_rejects_empty_numeric_payload(tmp_path):
    backend = AnalogLifBackend(_load_synthetic_bundle(tmp_path))

    with pytest.raises(AnalogLifError, match="at least one neuron"):
        backend.execute(ipos_a=[], ineg_a=[], vmem0_v=[])

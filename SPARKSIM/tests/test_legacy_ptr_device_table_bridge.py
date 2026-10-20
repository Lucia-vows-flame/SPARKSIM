import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from MNSIM.Scheduling.phase2.physical_write_ledger import PhysicalWriteTables


def test_legacy_adapter_preserves_canonical_costs_units_and_config_binding(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/generate_legacy_flashsef_ptr.py"
    spec = importlib.util.spec_from_file_location("legacy_generator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = PhysicalWriteTables(sim_config_path="canonical.ini", xbar_shape=(128, 128), ou_shape=(4, 4), device_level=256,
                                latency_ns=np.array([[0., 995.3975297362646], [100., 0.]]),
                                energy_nj=np.array([[0., 1.349101829448027], [.28956516475352323, 0.]]),
                                table_digest="authoritative")
    physical_module = SimpleNamespace(PhysicalWriteTables=SimpleNamespace)
    config = tmp_path / "frozen.ini"
    bridged = module.install_transition_table_bridge(physical_module, reference, config)
    assert physical_module.load_physical_write_tables(str(config)) is bridged
    assert bridged.table_digest == reference.table_digest
    np.testing.assert_array_equal(bridged.latency_ns, reference.latency_ns)
    np.testing.assert_array_equal(bridged.energy_nj, reference.energy_nj)
    assert not np.shares_memory(bridged.energy_nj, reference.energy_nj)
    with pytest.raises(ValueError, match="unfrozen"):
        physical_module.load_physical_write_tables(tmp_path / "different.ini")
    with pytest.raises(ValueError):
        bridged.energy_nj[0, 1] = 0

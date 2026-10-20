"""Round-orchestrated interface facade.

It delegates model construction and execution to the original MNSIM
``TrainTestInterface``.  The only added behavior is checkpoint preparation
and selection of strict accuracy dataset adapters.
"""

from __future__ import annotations

from MNSIM.Interface.interface import TrainTestInterface

from round_orchestrated_compat import load_network_weights_compat


class RoundOrchestratedTrainTestInterface(TrainTestInterface):
    """Use the original interface while adapting legacy input artifacts."""

    def __init__(self, *args, **kwargs):
        weights_file = kwargs.get("weights_file")
        use_prequantized = bool(kwargs.get("use_prequantized_weights", False))
        if weights_file is None or use_prequantized:
            super().__init__(*args, **kwargs)
            return

        kwargs["weights_file"] = None
        super().__init__(*args, **kwargs)
        self.weights_file = weights_file
        load_network_weights_compat(self.net, weights_file, map_location=self.device)

    def _get_calibration_loader(self):
        if self.dataset_module == "round_orchestrated_snn_dataset":
            from round_orchestrated_snn_dataset import get_calibration_dataloader

            return get_calibration_dataloader()
        return super()._get_calibration_loader()

    def test_split_identity(self):
        if self.dataset_module == "round_orchestrated_snn_dataset":
            from round_orchestrated_snn_dataset import get_dataloader

            if self.test_loader is None:
                self.test_loader = get_dataloader()[1]
            return self.test_loader.dataset.accuracy_split_identity()
        return super().test_split_identity()

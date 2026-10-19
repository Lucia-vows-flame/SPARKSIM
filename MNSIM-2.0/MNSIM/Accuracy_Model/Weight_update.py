import sys
import os
import math
import random
import configparser as cp
import numpy as np
from MNSIM.Accuracy_Model.device_nonidealities import (
    DeviceNonidealityConfig,
    apply_cell_nonidealities,
)
from MNSIM.Interface.final_rram_contract import load_final_rram_model
work_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Interface.interface import *

def weight_update(
    SimConfig_path, weight, is_SAF=0, is_Variation=0, is_Rratio=0, rng=None
):
    # print("Hardware config file is loaded:", SimConfig_path)
    wu_config = cp.ConfigParser()
    wu_config.read(SimConfig_path, encoding='UTF-8')
    
    SAF_dist = list(map(float, wu_config.get('Device level', 'Device_SAF').split(',')))
    variation = float(wu_config.get('Device level', 'Device_Variation'))
    device_level = int(wu_config.get('Device level', 'Device_Level'))
    assert device_level >= 0, "NVM resistance level < 0"
    device_resistance = np.array(list(map(float, wu_config.get('Device level', 'Device_Resistance').split(','))))
    assert device_level == len(device_resistance), "NVM resistance setting error"
    variation_model = "multiplicative_gaussian_resistance"
    programming_error_slope = None
    programming_error_intercept_us = None
    if wu_config.has_section("Final RRAM Model"):
        final_model = load_final_rram_model(
            wu_config.get("Final RRAM Model", "Model_ID")
        )
        variation_config = final_model.get("variation", {})
        variation_model = str(
            variation_config.get("model", variation_model)
        )
        programming_error = variation_config.get("programming_error") or {}
        programming_error_slope = programming_error.get("slope")
        programming_error_intercept_us = programming_error.get("intercept_us")
    nonidealities = DeviceNonidealityConfig(
        resistances_ohm=tuple(device_resistance),
        variation_percent=variation,
        saf_hrs_percent=SAF_dist[0],
        saf_lrs_percent=SAF_dist[1],
        variation_model=variation_model,
        programming_error_slope=programming_error_slope,
        programming_error_intercept_us=programming_error_intercept_us,
    )
    if rng is None:
        # Retain compatibility with callers that scope np.random.seed.
        class _LegacyGenerator:
            normal = staticmethod(np.random.normal)
            random = staticmethod(np.random.random_sample)
        rng = _LegacyGenerator()
    elif not hasattr(rng, "normal") or not hasattr(rng, "random"):
        raise TypeError("rng must provide normal() and random()")
    for i in range(len(weight)):
        if weight[i] is not None:
            for label, value in weight[i].items():
                # print(value.shape)
                value = apply_cell_nonidealities(
                    value,
                    nonidealities,
                    enable_r_ratio=bool(is_Rratio),
                    enable_variation=bool(is_Variation),
                    enable_saf=bool(is_SAF),
                    rng=rng,
                )
                weight[i].update({label: value.astype(float)})
    # print(total_size, total_difference/total_size)
    #    print true error rate
    return weight

if __name__ == '__main__':
    SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"cifar10_vgg8_params.pth")
    __TestInterface = TrainTestInterface('vgg8', 'MNSIM.Interface.cifar10', SimConfig_path,
                                         weights_file_path, 0)
    structure_file = __TestInterface.get_structure()
    weight = __TestInterface.get_net_bits()
    weight_2 = weight_update(SimConfig_path, weight, is_Variation=0,is_SAF=1,is_Rratio=0)
    weight = __TestInterface.get_net_bits()
    # print(__TestInterface.set_net_bits_evaluate(weight_2))







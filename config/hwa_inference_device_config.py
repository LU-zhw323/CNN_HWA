import dataclasses
from typing import List

@dataclasses.dataclass
class DeviceConfig:
    name: str
    drift_scale: float
    g_min: float
    g_max: float

@dataclasses.dataclass
class CNN_HWA_INFERENCE_DEVICE_Config:
    # model parameters
    batch_size: int = 50
    epochs: int = 600
    fp_error: float = 0.05879999999999996

    # hwa training parameters
    initial_hwa_noise_scale: float = 0.0
    hwa_noise_scale: float = 3.0
    pdrop: float = 0.00
    lr: float = 7.5e-3
    lr_decay_factor: float = 0.1 # applied after each epoch if valid loss not improved
    lr_milestones: List[int] = dataclasses.field(default_factory=lambda: [300, 500])
    momentum: float = 0.9
    max_grad_norm: float = None
    weight_decay: float = 1e-3

    
    # noise model parameters
    base_drift_coeff: float = 0.049
    noise_scale: List[float] = dataclasses.field(default_factory=lambda: [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    devices: List[DeviceConfig] = dataclasses.field(default_factory=lambda: [
        DeviceConfig(name='liner_1', drift_scale=0.01/0.049, g_min=0.5, g_max=20.0),
        DeviceConfig(name='liner_2', drift_scale=0.02/0.049, g_min=1.0, g_max=300.0),
        DeviceConfig(name='homo', drift_scale=0.025/0.049, g_min=1.0, g_max=20.0),
        DeviceConfig(name='opt', drift_scale=0.04/0.049, g_min=0.0, g_max=75.0),
        DeviceConfig(name='cpl', drift_scale=0.03/0.049, g_min=0.1, g_max=100.0),
    ])
    # hwa evaluation parameters
    num_evals: int = 25
    inference_time: List[float] = dataclasses.field(default_factory=lambda: [1, 3600, 3600*24, 3600*24*7, 3600*24*365])


    
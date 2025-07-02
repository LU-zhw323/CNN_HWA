import dataclasses
from typing import List



@dataclasses.dataclass
class CNN_HWA_INFERENCE_Config:
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
    noise_scale: List[float] = dataclasses.field(default_factory=lambda: [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    drift_scale: List[float] = dataclasses.field(default_factory=lambda: [0.005, 0.05, 0.5, 1.0])
    g_min: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.005, 0.05, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    g_max: float = 25.0

    # hwa evaluation parameters
    num_evals: int = 25
    inference_time: List[float] = dataclasses.field(default_factory=lambda: [1, 3600, 3600*24, 3600*24*7, 3600*24*365])


    
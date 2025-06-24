import dataclasses
from typing import List



@dataclasses.dataclass
class CNN_HWA_Config:
    # model parameters
    batch_size: int = 50
    epochs: int = 600

    # hwa training parameters
    initial_hwa_noise_scale: float = 1e-3
    hwa_noise_scale: float = 3.0
    ramp_up_ratio: float = 0.1 # ramp up noise in the first 60 epochs
    pdrop: float = 0.00
    lr: float = 7.5e-3
    lr_decay_factor: float = 0.1 # applied after each epoch if valid loss not improved
    lr_milestones: List[int] = dataclasses.field(default_factory=lambda: [300, 500])
    momentum: float = 0.9
    max_grad_norm: float = None
    weight_decay: float = 1e-3

    # noise model parameters
    noise_scale: float = 1.0
    drift_scale: float = 1.0
    g_min: float = 0.0
    g_max: float = 25.0

    # hwa evaluation parameters
    num_evals: int = 3
    t_inference: float = 365 * 24 * 60 * 60 # 1 year


    
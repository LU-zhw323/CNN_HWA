import dataclasses
from typing import List



@dataclasses.dataclass
class CNN_HWA_INFERENCE_Config:
    """FP baseline and sweep grid of `hwa_inference.py`."""

    # fp baseline
    fp_error: float = 0.05879999999999996
    """Test error rate of the FP model, fraction. The 100% point of the normalized accuracy."""

    # hwa training parameters
    hwa_noise_scale: float = 3.0
    """Std of the PCM weight-noise modifier in the RPU config. Inactive in evaluation."""

    # noise model parameters
    noise_scale: List[float] = dataclasses.field(default_factory=lambda: [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    """Multipliers of the PCM programming and read noise of the aihwkit PCM model. Unitless."""
    drift_scale: List[float] = dataclasses.field(default_factory=lambda: [0.05, 0.5, 1.0])
    """Multipliers of the PCM drift coefficient of the aihwkit PCM model. Unitless."""
    g_min: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.005, 0.05, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    """Minimum device conductances, uS. The memory window is g_max - g_min."""
    g_max: float = 25.0
    """Maximum device conductance, uS."""

    # hwa evaluation parameters
    num_evals: int = 25
    """Evaluations per configuration, averaged. All share one programming-noise draw; each draws new read noise."""
    inference_time: List[float] = dataclasses.field(default_factory=lambda: [1, 3600, 3600*24, 3600*24*7, 3600*24*365])
    """Times after programming at which the weights are drifted, s."""

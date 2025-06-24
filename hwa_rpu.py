from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    BoundManagementType,
    WeightClipType,
    NoiseManagementType,
    WeightRemapType,
    WeightNoiseType,
)
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.simulator.presets.utils import IOParameters

def hwa_rpu_config(
        hwa_noise_scale: float=3.0,
        noise_scale: float=1.0, 
        drift_scale: float=1.0, 
        g_min: float=0.0, 
        g_max: float=25.0
    )->InferenceRPUConfig:
    """
    Generate a rpu config for HWA training
    Args:
        hwa_noise_scale: noise scale for hwa training
        hwa_pdrop: dropout rate for hwa training
        noise_scale: noise scale for noise model
        drift_scale: drift scale for noise model
        g_min: minimum conductance for noise model
        g_max: maximum conductance for noise model
    Returns:
        rpu_config: a rpu config for hwa training
    """
    rpu_config = InferenceRPUConfig()

    # modifier for hwa training
    rpu_config.modifier.type = WeightModifierType.PCM_NOISE
    rpu_config.modifier.std_dev = hwa_noise_scale
    rpu_config.modifier.pcm_t0 = 20.0

    # weight clipping
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.5

    # forward
    rpu_config.forward.out_res = 1.0 / (2**8 - 2)
    rpu_config.forward.inp_res = 1.0 / (2**8 - 2)
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.out_bound = 10.0 #1.0 if rpu_config.mapping.learn_out_scaling = True
    #rpu_config.forward.ir_drop_g_ratio = 571428.57
    rpu_config.forward.ir_drop = 1.0
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE
    rpu_config.forward.w_noise_type = WeightNoiseType.PCM_READ
    rpu_config.forward.w_noise = 0.0175
    rpu_config.forward.inp_bound = 1.0

    
    # mapping
    rpu_config.mapping.max_input_size = 512
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    #rpu_config.mapping.weight_scaling_lr_compensation = True
    

    # learn input range
    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.learn_input_range = True
    rpu_config.pre_post.input_range.decay = 0.001
    rpu_config.pre_post.input_range.gradient_relative = True
    rpu_config.pre_post.input_range.gradient_scale = 1.0
    rpu_config.pre_post.input_range.init_from_data = 100
    

    # noise model
    rpu_config.noise_model = PCMLikeNoiseModel(
        g_max=g_max,
        prog_noise_scale=noise_scale,
        read_noise_scale=noise_scale,
        drift_scale=drift_scale,
        g_converter=SinglePairConductanceConverter(g_max=g_max, g_min=g_min),
    )
    rpu_config.drift_compensation = GlobalDriftCompensation()

    return rpu_config





def pretain_rpu_config():
    """
    Generate a rpu config for the aihwkit pretained model
    """
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.std_dev = 0.06
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0

    rpu_config.forward = IOParameters()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1 / (2**8 - 2)
    rpu_config.forward.out_bound = 10
    rpu_config.forward.out_res = 1 / (2**8 - 2)
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.decay = 0.01
    rpu_config.pre_post.input_range.init_from_data = 50
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.input_min_percentage = 0.995
    rpu_config.pre_post.input_range.manage_output_clipping = False

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config
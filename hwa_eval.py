import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from tqdm import tqdm
from hwa_utils import covert_fp_to_hwa, evaluate_hwa, inference_hwa, load_hwa_model, ramp_up_noise, save_hwa_model, train_step_hwa, load_hwa_model
from resnet import resnet32
from hwa_rpu import hwa_rpu_config
from config import CNN_HWA_Config
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from utils import evaluate_fp, set_seed, create_two_step_lr_schedule
from data import load_cifar10_data

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
from utils import compute_norm_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_cnn.th"
HWA_CHECKPOINT_PATH = "checkpoints/fine.th"

def gen_rpu_config():
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


def main():
    _, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)

    model = resnet32().to(DEVICE)
    rpu_config = gen_rpu_config()
    hwa_model = convert_to_analog(model, rpu_config).to(DEVICE)
    hwa_model.load_state_dict(torch.load(HWA_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False))
    t_inference = 365 * 24 * 60 * 60
    test_loss, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, test_data, t_inference, 3, DEVICE)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}, Test Error Rate: {test_error_rate:.3f}")

    norm_acc = compute_norm_accuracy(0.05879999999999996, test_error_rate, 10)
    print(f"Normal Accuracy: {norm_acc:.3f}")

if __name__ == "__main__":
    main()
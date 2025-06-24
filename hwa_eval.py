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
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"




def main():
    _, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=3,
        noise_scale=1,
        drift_scale=1,
        g_min=0,
        g_max=25.0
    )

    hwa_model = load_hwa_model(HWA_CHECKPOINT_PATH, rpu_config, DEVICE, True)
    t_inference = 365 * 24 * 60 * 60
    test_loss, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, test_data, t_inference, 3, DEVICE)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}, Test Error Rate: {test_error_rate:.3f}")

    norm_acc = compute_norm_accuracy(0.05879999999999996, test_error_rate, 10)
    print(f"Normal Accuracy: {norm_acc:.3f}")

if __name__ == "__main__":
    main()
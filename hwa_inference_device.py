import fcntl
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
from config import CNN_HWA_INFERENCE_DEVICE_Config
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
import os
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_cnn.th"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"
RESULTS_DIR = "results"





def create_csv_writer(csv_path):
    """
    Create a CSV writer for the given path.
    If the file exists, open it in append mode.
    If the file does not exist, create it and write the header.
    """
    file_exists = os.path.exists(csv_path)
    
    if file_exists:
        file_obj = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(file_obj)
    else:
        # create new file and write header
        file_obj = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(file_obj)
        
        header = ['t_inference', 'device', 'noise_scale', 'drift_scale', 'g_min', 'g_max', 
                  'memory_window', 'loss', 'accuracy', 'error', 'norm_accuracy', 'norm_error']
        csv_writer.writerow(header)
    
    return file_obj, csv_writer, file_exists



def save_experiment_result(csv_writer, t_inference, device, noise_scale, drift_scale, 
                          g_min, g_max, loss, accuracy, error_rate, norm_accuracy, norm_error):
    """
    Save an experiment result to a CSV file.
    
    Args:
        csv_writer: CSV writer
        t_inference: inference time
        device: device
        noise_scale: noise scale
        drift_scale: drift scale
        g_min: minimum conductance
        g_max: maximum conductance
        loss: loss
        perplexity: perplexity
        accuracy: accuracy
        error_rate: error rate
        norm_accuracy: normalized accuracy
        norm_error: normalized error
    """
    memory_window = g_max - g_min
    row = [t_inference, device, noise_scale, drift_scale, g_min, g_max, 
           memory_window, loss, accuracy, error_rate, norm_accuracy, norm_error]
    csv_writer.writerow(row)





def main():
    # setup rpu config
    cnn_hwa_inference_device_config = CNN_HWA_INFERENCE_DEVICE_Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # baseline
    fp_error = cnn_hwa_inference_device_config.fp_error


    # load data
    _, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)

    # time label
    time_mapping = {
        1: 'second',
        3600: 'hour',
        3600 * 24: 'day',
        3600 * 24 * 7: 'week',
        3600 * 24 * 365: 'year'
    }
    total_all_experiments = len(cnn_hwa_inference_device_config.inference_time) * len(cnn_hwa_inference_device_config.noise_scale) * \
                           len(cnn_hwa_inference_device_config.devices)
    print(f"\nTotal experiments to run: {total_all_experiments}")
    
    # inference

    for device in cnn_hwa_inference_device_config.devices:
        device_name = device.name
        drift_scale = device.drift_scale
        g_min = device.g_min
        g_max = device.g_max
        
        for t_inference in cnn_hwa_inference_device_config.inference_time:
            t_label = time_mapping[t_inference]
            csv_filename = f"device_results_{t_label}.csv"
            csv_path = os.path.join(RESULTS_DIR, csv_filename)
            file_obj, csv_writer, file_exists = create_csv_writer(csv_path)
            print(f"\n{'Appending to' if file_exists else 'Creating'} {csv_filename} (t={t_inference}s)")

            total_experiments = len(cnn_hwa_inference_device_config.noise_scale)
            
            pbar = tqdm(total=total_experiments, 
                        desc=f"Device={device_name.upper()}, T={t_label.upper()} experiments")
            
            try:
                for noise_scale in cnn_hwa_inference_device_config.noise_scale:
                    # update progress bar description
                    pbar.set_description(f"Device={device_name.upper()}, T={t_label.upper()}, noise={noise_scale}")

                    # set rpu config
                    rpu_config = hwa_rpu_config(
                        hwa_noise_scale=cnn_hwa_inference_device_config.hwa_noise_scale,
                        noise_scale=noise_scale,
                        drift_scale=drift_scale,
                        g_min=g_min,
                        g_max=g_max,
                    )

                    # load model
                    hwa_model = load_hwa_model(HWA_CHECKPOINT_PATH, rpu_config, DEVICE, False)
                    
                    # evaluate hwa model
                    test_loss, test_accuracy, test_error_rate = inference_hwa(
                        hwa_model, test_data, t_inference, cnn_hwa_inference_device_config.num_evals, DEVICE)
                    
                        # get normalized error rate
                    norm_accuracy = compute_norm_accuracy(fp_error, test_error_rate, 10)
                    norm_error = 1.0 - norm_accuracy
                    
                    # save results to CSV
                    save_experiment_result(csv_writer, t_inference, device_name, noise_scale, 
                                            drift_scale, g_min, g_max, test_loss, 
                                            test_accuracy, test_error_rate, norm_accuracy, norm_error)
                    
                    # flush file buffer to ensure data is written
                    file_obj.flush()
                    
                    # update progress bar
                    pbar.update(1)
                    
            finally:
                pbar.close()
                file_obj.close()
                print(f"\nResults saved to {csv_path}")
            
            
    
    print("\nAll experiments completed!")
                    
        



if __name__ == "__main__":
    main()
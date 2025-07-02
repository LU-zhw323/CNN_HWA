import csv
import math
import os
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import numpy as np
from tqdm import tqdm
from data import load_cifar10_data
from hwa_utils import covert_fp_to_hwa, evaluate_hwa, inference_hwa, load_hwa_model, save_hwa_model, train_step_hwa
from resnet import resnet32
from hwa_rpu import hwa_rpu_config
from config import CNN_HWA_INFERENCE_Config
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from utils import set_seed
from utils import compute_norm_accuracy
import argparse
import fcntl
DATA_PATH = "data/ptb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_cnn.th"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"
RESULTS_DIR = "results"





def create_csv_writer(csv_path):
    file_exists = os.path.exists(csv_path)
    
    file_obj = open(csv_path, 'a' if file_exists else 'w', newline='')
    
    fcntl.flock(file_obj, fcntl.LOCK_EX)
    try:
        csv_writer = csv.writer(file_obj)
        
        if not file_exists:
            header = ['t_inference', 'noise_scale', 'drift_scale', 'g_min', 'g_max', 
                      'memory_window', 'loss', 'accuracy', 'error', 'norm_accuracy', 'norm_error']
            csv_writer.writerow(header)
            file_obj.flush()
    finally:
        fcntl.flock(file_obj, fcntl.LOCK_UN)
    
    return file_obj, csv_writer, file_exists



def save_experiment_result(file_obj, csv_writer, t_inference, noise_scale, drift_scale, 
                          g_min, g_max, loss, accuracy, error_rate, norm_accuracy, norm_error):
    """
    Save an experiment result to a CSV file.
    
    Args:
        csv_writer: CSV writer
        t_inference: inference time
        noise_scale: noise scale
        drift_scale: drift scale
        g_min: minimum conductance
        g_max: maximum conductance
        loss: loss
        accuracy: accuracy
        error_rate: error rate
        norm_accuracy: normalized accuracy
        norm_error: normalized error
    """
    memory_window = g_max - g_min
    row = [t_inference, noise_scale, drift_scale, g_min, g_max, 
           memory_window, loss, accuracy, error_rate, norm_accuracy, norm_error]
    fcntl.flock(file_obj, fcntl.LOCK_EX)
    try:
        csv_writer.writerow(row)
        file_obj.flush()
    finally:
        fcntl.flock(file_obj, fcntl.LOCK_UN)


def get_param_combo(job_id, noise_scales, drift_scales, g_mins):
    N1 = len(noise_scales)
    N2 = len(drift_scales)
    N3 = len(g_mins)
    total = N1 * N2 * N3

    assert 0 <= job_id < total, f"job_id {job_id} out of range (0 to {total - 1})"

    i = job_id // (N2 * N3)
    j = (job_id % (N2 * N3)) // N3
    k = job_id % N3

    return noise_scales[i], drift_scales[j], g_mins[k]

def parse_args():
    parser = argparse.ArgumentParser(description="Run single HWA experiment by job id")
    parser.add_argument("--job_id", type=int, required=True, help="Job ID in range [1, total_combinations]")
    parser.add_argument("--t_inference", type=str, default="second", required=True, help="Inference time (in seconds, e.g., 1, 3600, etc.)")
    return parser.parse_args()



def main():
   
    # setup rpu config
    cnn_config = CNN_HWA_INFERENCE_Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)


    # get args
    args = parse_args()
    job_id = args.job_id - 1
    t_label = args.t_inference
    
    # baseline
    fp_error = cnn_config.fp_error

    # gmax
    g_max = cnn_config.g_max

    # load data
    _, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)

    # get number of batches
    num_test_batches = len(test_data)

    # time label
    time_mapping = {
        'second': 1,
        'hour': 3600,
        'day': 3600 * 24,
        'week': 3600 * 24 * 7,
        'year': 3600 * 24 * 365
    }

    # get param combo
    noise_scale, drift_scale, g_min = get_param_combo(job_id, cnn_config.noise_scale, cnn_config.drift_scale, cnn_config.g_min)

    # get t_label
    t_inference = time_mapping[t_label]

    # get csv path
    csv_filename = f"inference_results_{t_label}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    file_obj, csv_writer, file_exists = create_csv_writer(csv_path)


    # set rpu config
    rpu_config = hwa_rpu_config(
                    hwa_noise_scale=cnn_config.hwa_noise_scale,
                    noise_scale=noise_scale,
                    drift_scale=drift_scale,
                    g_min=g_min,
                    g_max=g_max,
                )
    # load model
    hwa_model = load_hwa_model(HWA_CHECKPOINT_PATH, rpu_config, DEVICE, False)
    
    # evaluate hwa model
    test_loss, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, test_data, t_inference, cnn_config.num_evals, DEVICE)
    
        # get normalized error rate
    norm_accuracy = compute_norm_accuracy(fp_error, test_error_rate, 10)
    norm_error = 1.0 - norm_accuracy
    
    # save results to CSV
    save_experiment_result(file_obj, csv_writer, t_inference, noise_scale, 
                            drift_scale, g_min, g_max, test_loss, 
                            test_accuracy, test_error_rate, norm_accuracy, norm_error)
    
    print(f"Inference Time={t_inference}s, Noise={noise_scale}, Drift={drift_scale}, g_min={g_min}, g_max={g_max}, "
          f"Loss={test_loss:.4f}, "
          f"Accuracy={test_accuracy:.4f}, Error={test_error_rate:.4f}, "
          f"Norm Accuracy={norm_accuracy:.4f}, Norm Error={norm_error:.4f}")
    
    # close file
    file_obj.close()
                    
        



if __name__ == "__main__":
    main()
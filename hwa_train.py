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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_CHECKPOINT_PATH = "checkpoints/fp_cnn.th"
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model.th"
HWA_FINAL_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"




def main():
    # set seed
    set_seed(42)
    # setup rpu config
    cnn_config = CNN_HWA_Config()
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=cnn_config.hwa_noise_scale,
        noise_scale=cnn_config.noise_scale,
        drift_scale=cnn_config.drift_scale,
        g_min=cnn_config.g_min,
        g_max=cnn_config.g_max
    )

    train_data, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)

    # get number of batches
    num_train_batches = len(train_data)
    num_test_batches = len(test_data)

    # load model
    model = resnet32().to(DEVICE)
    model.load_state_dict(torch.load(FP_CHECKPOINT_PATH))


    # convert fp model to hwa model
    hwa_model = covert_fp_to_hwa(model, rpu_config, DEVICE)

    # optimizer
    optimizer = AnalogSGD(hwa_model.parameters(), lr=cnn_config.lr, momentum=cnn_config.momentum, weight_decay=cnn_config.weight_decay)
    scheduler = create_two_step_lr_schedule(
        optimizer, start_lr=cnn_config.lr, 
        milestones=cnn_config.lr_milestones, 
        lr_decay_factor=cnn_config.lr_decay_factor,
        warmup_epochs=2
    )
    
    # train hwa model
    best_test_error_rate = float('inf')

    # count batches
    batch_count = 0


    for epoch in tqdm(range(cnn_config.epochs), desc="Training"):
        current_lr = optimizer.param_groups[0]['lr']

        # ramp up noise
        ramp_up_noise(batch_count, rpu_config)
        hwa_model.replace_rpu_config(rpu_config)

        # train hwa model
        train_loss, train_accuracy = train_step_hwa(
            hwa_model, rpu_config, train_data, optimizer, DEVICE)
        batch_count += num_train_batches
        print("-" * 80)
        print(f"Epoch {epoch+1:2d} | Lr: {current_lr:.3f} | Noise: {rpu_config.modifier.std_dev:.3f} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}")
        # evaluate hwa model
        test_loss, test_accuracy, test_error_rate = evaluate_hwa(
            hwa_model, test_data, cnn_config.num_evals, DEVICE)
        
        print(f"Epoch {epoch+1:2d} | Lr: {current_lr:.3f} | Noise: {rpu_config.modifier.std_dev:.3f} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
        print("-" * 80)
        if test_error_rate < best_test_error_rate:
            best_test_error_rate = test_error_rate
            save_hwa_model(hwa_model, HWA_CHECKPOINT_PATH)
        scheduler.step()

    # save final hwa model
    save_hwa_model(hwa_model, HWA_FINAL_CHECKPOINT_PATH)
    test_loss, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, test_data, cnn_config.t_inference, cnn_config.num_evals, DEVICE)
    print("-" * 80)
    print(f"FINAL | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
    print("-" * 80)

    # load best hwa model
    hwa_model = load_hwa_model(HWA_CHECKPOINT_PATH, rpu_config, DEVICE, True)
    # evaluate hwa model
    test_loss, test_accuracy, test_error_rate = inference_hwa(
        hwa_model, test_data, cnn_config.t_inference, cnn_config.num_evals, DEVICE)
    print("-" * 80)
    print(f"BEST | Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy:.3f} | Test Error Rate: {test_error_rate:.3f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
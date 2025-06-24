from typing import Tuple
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
import math
import numpy as np
import torch
import torch.nn.functional as F
from aihwkit.nn.conversion import convert_to_analog
import torch.nn as nn

from resnet import resnet32
from torch.serialization import add_safe_globals
from aihwkit.optim import AnalogSGD


def train_step_hwa(model, rpu_config, train_data, optimizer, device)->float:
    """
    Train the model on the data loader for hwa training
    Args:
        model: the model to train
        rpu_config: the rpu config to use
        train_data: the data loader to train on
        device: the device to train on
        optimizer: the optimizer to use
    Returns:
        avg_train_loss: the average loss
        avg_train_accuracy: the average accuracy
    """
    model.train()
    train_loss = 0.0
    correct = 0.0
    total_predictions = 0.0
    num_batches = len(train_data)
    
    for i, (images, labels) in enumerate(train_data, 0):
        # remap weights per 500 batches
        if (i + 1) % 500 == 0:
            model.remap_analog_weights()

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        # update weights
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    avg_train_loss = train_loss / num_batches
    avg_train_accuracy = correct / total_predictions
    return avg_train_loss, avg_train_accuracy


@torch.no_grad()
def evaluate_hwa(model, data_loader, num_evals, device):
    """
    Evaluate the model on the data loader for hwa training
    Args:
        model: the model to evaluate
        data_loader: the data loader to evaluate on
        num_evals: the number of evaluations
        device: the device to evaluate on
    Returns:
        avg_loss: the average loss
        accuracy: the accuracy
        error_rate: the error rate
    """
    model.eval()

    all_losses = []
    all_accuracies = []
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for _ in range(num_evals):
            total_loss = 0.0
            total_correct = 0.0
            total_predictions = 0.0
            for i, (images, labels) in enumerate(data_loader, 0):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            trial_loss = total_loss / num_batches
            trial_accuracy = total_correct / total_predictions
            all_losses.append(trial_loss)
            all_accuracies.append(trial_accuracy)
    
    avg_loss = np.mean(all_losses)
    avg_accuracy = np.mean(all_accuracies)
    avg_error_rate = 1 - avg_accuracy
    return avg_loss, avg_accuracy, avg_error_rate


@torch.no_grad()
def inference_hwa(model, data_loader, t_inference, num_evals, device):
    """
    Evaluate the model on the data loader for hwa inference
    Args:
        model: the model to evaluate
        data_loader: the data loader to evaluate on
        t_inference: the time of inference
        num_evals: the number of evaluations
        device: the device to evaluate on
    Returns:
        avg_loss: the average loss
        perplexity: the perplexity
        accuracy: the accuracy
        error_rate: the error rate
    """
    model.eval()
    all_losses = []
    all_accuracies = []
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for _ in range(num_evals):
            total_loss = 0.0
            total_correct = 0.0
            total_predictions = 0.0
            # drift analog weights
            model.drift_analog_weights(t_inference)
            for i, (images, labels) in enumerate(data_loader, 0):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            trial_loss = total_loss / num_batches
            trial_accuracy = total_correct / total_predictions
            all_losses.append(trial_loss)
            all_accuracies.append(trial_accuracy)
    
    avg_loss = np.mean(all_losses)
    avg_accuracy = np.mean(all_accuracies)
    avg_error_rate = 1 - avg_accuracy
    return avg_loss, avg_accuracy, avg_error_rate




def covert_fp_to_hwa(fp_model: nn.Module, rpu_config: InferenceRPUConfig, device: torch.device)->nn.Module:
    """
    Convert the fp model to a hwa model
    Args:
        fp_model: the fp model to convert
        rpu_config: the rpu config to use
        device: the device to convert the model to
    Returns:
        hwa_model: the hwa model
    """
    # convert fp model to hwa model
    hwa_model = convert_to_analog(fp_model, rpu_config).to(device)
    return hwa_model
   



def save_hwa_model(analog_model, analog_model_path):
    """
    Save the hwa model
    Args:
        analog_model: the hwa model to save
        encoder: the encoder to save
        filepath: the path to save the model
    """
    torch.save(analog_model.state_dict(), analog_model_path)


def load_hwa_model(analog_model_path, rpu_config, device, load_rpu=False)->nn.Module:
    """
    Load the saved HWA model and encoder
    
    Args:
        analog_model_path: path to the saved analog model
        rpu_config: RPU configuration for analog conversion
        device: torch device to load models onto
        
    Returns:
        hwa_model: loaded analog model
    """
    model = resnet32().to(device)

    hwa_model = convert_to_analog(model, rpu_config).to(device)
    hwa_model.load_state_dict(torch.load(analog_model_path, map_location=device, weights_only=False), load_rpu_config=load_rpu)
    return hwa_model


def ramp_up_noise(batch_count, rpu_config, max_batches=20000, initial_noise=0.0, max_noise=3.0):

    # ramp up noise in the first 100 epochs
    noise_ramp_ratio = min(batch_count / max_batches, 1.0)
    current_noise = initial_noise + (max_noise - initial_noise) * noise_ramp_ratio

    # update rpu config
    rpu_config.modifier.std_dev = current_noise



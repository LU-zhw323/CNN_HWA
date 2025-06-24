import random
import numpy as np
import torch
from torch.nn import functional as F
import math

def set_seed(seed=42):
    random.seed(seed)                     
    np.random.seed(seed)                  
    torch.manual_seed(seed)               
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     



@torch.no_grad()
def evaluate_fp(model, data_loader, device):
    """
    Evaluate the model on the data loader for fp training
    Args:
        model: the model to evaluate
        data_loader: the data loader to evaluate on
        device: the device to evaluate on
    Returns:
        avg_loss: the average loss
        accuracy: the accuracy
        error_rate: the error rate
    """
    
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader, 0):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_predictions
    error_rate = 1 - accuracy
    return avg_loss, accuracy, error_rate





def compute_norm_accuracy(fp_error: float, hwa_error: float, num_classes: int = 10):
    # compute chance error
    error_chance = 1.0 - 1.0 / num_classes

    # compute norm error
    return  1.0 - (hwa_error - fp_error) / (error_chance - fp_error)



def create_two_step_lr_schedule(optimizer, start_lr=7.5e-3, warmup_epochs=2, milestones=[300, 500], 
                               final_lr_ratio=0.01, lr_decay_factor=0.1,
                               warmup_reduction=20):
    """
    2 step lr schedule
    
    Args:
        optimizer: optimizer
        start_lr: start learning rate
        warmup_epochs: warmup epochs
        total_epochs: total epochs
        final_lr_ratio: final learning rate ratio
        warmup_reduction: warmup learning rate reduction factor
    """
    
    warmup_lr = start_lr / warmup_reduction  # 0.025 / 20 = 0.00125
    final_lr = start_lr * final_lr_ratio     # 0.025 * 0.01 = 0.00025
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # warmup
            progress = epoch / warmup_epochs
            current_lr = warmup_lr + (start_lr - warmup_lr) * progress
            return current_lr / start_lr
        elif epoch < milestones[0]:
            # first stage: keep start_lr
            return 1.0
        elif epoch < milestones[1]:
            # second stage: drop to final_lr and keep
            return lr_decay_factor
        else:
            # second stage: drop to final_lr and keep
            return final_lr / start_lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
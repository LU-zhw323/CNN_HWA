import torch
from hwa_utils import inference_hwa, load_hwa_model
from hwa_rpu import hwa_rpu_config
from data import load_cifar10_data
from utils import compute_norm_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HWA_CHECKPOINT_PATH = "checkpoints/hwa_model_final.th"




def main():
    """Prints test metrics of `HWA_CHECKPOINT_PATH` at 1 year, 1 week, 1 day, 1 hour and 1 s after programming.

    Uses the RPU config stored in the checkpoint. Programs the weights once, at the first time point,
    and averages 25 evaluations per time.
    """
    _, test_data = load_cifar10_data(batch_size=50, num_workers=2, use_augmentation=True)
    rpu_config = hwa_rpu_config(
        hwa_noise_scale=3,
        noise_scale=1,
        drift_scale=1,
        g_min=0,
        g_max=25.0
    )

    hwa_model = load_hwa_model(HWA_CHECKPOINT_PATH, rpu_config, DEVICE, True)
    t_map = {
        'year': 365 * 24 * 60 * 60,
        'week': 7 * 24 * 60 * 60,
        'day': 24 * 60 * 60,
        'hour': 60 * 60,
        'second': 1
    }
    for key, value in t_map.items():
        t_inference = value
        print(f"Evaluating {key}...")
        test_loss, test_accuracy, test_error_rate = inference_hwa(
            hwa_model, test_data, t_inference, 25, DEVICE)
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}, Test Error Rate: {test_error_rate:.3f}")

        norm_acc = compute_norm_accuracy(0.05879999999999996, test_error_rate, 10)
        print(f"Normal Accuracy: {norm_acc:.3f}")
        print("-"*100)

if __name__ == "__main__":
    main()

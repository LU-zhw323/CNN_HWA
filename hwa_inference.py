import os
import csv
import torch
from tqdm import tqdm

from data import load_cifar10_data
from hwa_utils import (
    inference_hwa,
    load_hwa_model,
)
from hwa_rpu import hwa_rpu_config
from config import CNN_HWA_INFERENCE_Config
from utils import compute_norm_accuracy


# ====================== PATHS & DEVICE ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HWA_CHECKPOINT_PATH = [
    "checkpoints/hwa_model_final_1.th",
    "checkpoints/hwa_model_final_2.th",
    "checkpoints/hwa_model_final_3.th",
    "checkpoints/hwa_model_final_4.th",
]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ====================== UTILS ======================
def load_existing_results(csv_path):
    """
    Load existing (t_inference, noise_scale, drift_scale, g_min) from CSV.
    Used to skip finished experiments.
    """
    done = set()
    if not os.path.exists(csv_path):
        return done

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                int(float(row["t_inference"])),
                round(float(row["noise_scale"]), 6),
                round(float(row["drift_scale"]), 6),
                round(float(row["g_min"]), 6),
            )
            done.add(key)
    return done


# ====================== MAIN ======================
def main():
    """Sweeps noise scale, drift scale, g_min and inference time for every checkpoint in `HWA_CHECKPOINT_PATH`.

    Writes one row per configuration to `RESULTS_DIR/inference_results_<time>_<k>.csv`, where k is the
    1-based position of the checkpoint in `HWA_CHECKPOINT_PATH`. Skips configurations already in the file.
    """
    cnn_config = CNN_HWA_INFERENCE_Config()

    # FP baseline
    fp_error = cnn_config.fp_error
    g_max = cnn_config.g_max

    # Load data ONCE
    _, test_data = load_cifar10_data(
        batch_size=50,
        num_workers=2,
        use_augmentation=True,
    )

    time_mapping = {
        1: "second",
        3600: "hour",
        3600 * 24: "day",
        3600 * 24 * 7: "week",
        3600 * 24 * 365: "year",
    }

    for ckpt_id, checkpoint_path in enumerate(HWA_CHECKPOINT_PATH):

        for t_inference in cnn_config.inference_time:
            t_label = time_mapping[t_inference]
            csv_name = f"inference_results_{t_label}_{ckpt_id + 1}.csv"
            csv_path = os.path.join(RESULTS_DIR, csv_name)

            # ---- load existing results
            done_set = load_existing_results(csv_path)

            file_exists = os.path.exists(csv_path)
            f = open(csv_path, "a", newline="")
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "t_inference",
                    "noise_scale",
                    "drift_scale",
                    "g_min",
                    "g_max",
                    "memory_window",
                    "loss",
                    "accuracy",
                    "error",
                    "norm_accuracy",
                    "norm_error",
                ])
                f.flush()

            # ---- collect remaining experiments
            remaining = []
            for noise_scale in cnn_config.noise_scale:
                for drift_scale in cnn_config.drift_scale:
                    for g_min in cnn_config.g_min:
                        key = (
                            t_inference,
                            round(noise_scale, 6),
                            round(drift_scale, 6),
                            round(g_min, 6),
                        )
                        if key not in done_set:
                            remaining.append((noise_scale, drift_scale, g_min))

            if len(remaining) == 0:
                print(f"[SKIP] {csv_name}: all experiments done")
                f.close()
                continue

            print(f"[RUN] {csv_name}: {len(remaining)} experiments remaining")

            pbar = tqdm(remaining, desc=f"T={t_label.upper()}")

            for noise_scale, drift_scale, g_min in pbar:
                pbar.set_postfix(
                    noise=noise_scale,
                    drift=drift_scale,
                    g_min=g_min,
                )

                rpu_config = hwa_rpu_config(
                    hwa_noise_scale=cnn_config.hwa_noise_scale,
                    noise_scale=noise_scale,
                    drift_scale=drift_scale,
                    g_min=g_min,
                    g_max=g_max,
                )

                # must reload model because RPU config changes
                hwa_model = load_hwa_model(
                    checkpoint_path,
                    rpu_config,
                    DEVICE,
                    False,
                )

                with torch.no_grad():
                    loss, acc, err = inference_hwa(
                        hwa_model,
                        test_data,
                        t_inference,
                        cnn_config.num_evals,
                        DEVICE,
                    )

                norm_acc = compute_norm_accuracy(fp_error, err, 10)
                norm_err = 1.0 - norm_acc

                writer.writerow([
                    t_inference,
                    noise_scale,
                    drift_scale,
                    g_min,
                    g_max,
                    g_max - g_min,
                    loss,
                    acc,
                    err,
                    norm_acc,
                    norm_err,
                ])
                f.flush()

            f.close()
            print(f"[DONE] Results saved to {csv_path}")

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()

# CNN HWA Training & Inference

This repository does hardware-aware (HWA) training and inference of ResNet-32 on CIFAR-10. Phase-change-memory (PCM) non-idealities are simulated with [IBM AIHWKIT](https://github.com/IBM/aihwkit). The training setup follows Rasch et al., *Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators*, Nature Communications (2023), [arXiv:2302.08469](https://arxiv.org/abs/2302.08469).

## Model

**FP model.** ResNet-32 from `resnet.py`:
- 3 stages of 5 basic blocks, with 16 / 32 / 64 channels and option-A shortcuts.
- `checkpoints/fp_cnn.th` reaches test accuracy 94.12% (error rate 0.0588, `fp_eval.py`).

**HWA model.** `aihwkit.nn.conversion.convert_to_analog` turns every convolution and linear layer into an AIHWKIT analog tile. Batch norm stays digital. `hwa_train.py` then trains it with the settings in `config/hwa_config.py`:
- 600 epochs, batch size 50.
- `AnalogSGD` with learning rate 7.5e-3, momentum 0.9, weight decay 1e-3.
- The learning rate is ×1/20 for the first 2 epochs, ×0.1 from epoch 300, and ×0.01 from epoch 500.
- The PCM weight-noise modifier rises linearly from 0 to std 3.0 over the first 20 epochs (20,000 batches).
- The analog weights are remapped every 500 batches.
- Augmentation: random crop, horizontal flip and 8-pixel cutout.

The tile configuration is in `hwa_rpu.hwa_rpu_config`:
- 8-bit input and output.
- Output noise 0.04.
- Layer-wise Gaussian weight clipping at 2.5 sigma.
- Learned input ranges and output scales, and column-wise weight scaling.

`hwa_train.py` writes two checkpoints:
- **Final:** the last epoch. All inference results use this one.
- **Best:** the epoch with the lowest **test** error. No result uses it.

**Inference.**
- The weights are programmed and drifted with AIHWKIT's `PCMLikeNoiseModel`, with global drift compensation.
- `noise_scale` scales the programming and read noise.
- `drift_scale` scales the drift.
- `g_min` and `g_max` (µS) set the conductance range. The memory window is `g_max - g_min`.
- The sweep loads the model fresh for every configuration.
- The first `drift_analog_weights(t)` call programs the weights. It draws the programming noise and the per-device drift exponents once.
- Each of the `num_evals` evaluations drifts these programmed weights to time t and draws new accumulated read noise. Output noise is drawn on every forward pass.
- Reported metrics average the `num_evals` evaluations.

**Normalized accuracy** (`utils.compute_norm_accuracy`):

```
A = 1 - (err_HWA - err_FP) / (err_chance - err_FP),   err_chance = 0.9,   err_FP = 0.0588
```

A is 1 at the FP error rate and 0 at chance. It can slightly exceed 1.

## Layout

| Path | Contents |
|---|---|
| `data.py` | CIFAR-10 loaders, with augmentation for training |
| `resnet.py` | ResNet-32 |
| `hwa_rpu.py` | AIHWKIT RPU configuration: training noise and the PCM inference model |
| `hwa_utils.py` | FP-to-analog conversion, HWA train/eval/inference loops, HWA checkpoint I/O, noise ramp-up |
| `utils.py` | Seeding, FP evaluation, normalized accuracy, learning-rate schedule |
| `config/` | Hyperparameters: `hwa_config.py`, `hwa_inference_config.py` |
| `fp_eval.py` | FP evaluation |
| `hwa_train.py`, `hwa_eval.py` | HWA training and evaluation of one model |
| `hwa_inference.py` | Inference sweep |
| `hpc.sh`, `hwa_inference_slurm.py` | Legacy. Do not use (see below) |
| `data/cifar-10-batches-py/` | CIFAR-10. `torchvision` downloads it again if it is missing |

## Environment

| Package | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.7.1+cu128 (CUDA 12.8) |
| torchvision | 0.22.1+cu128 |
| AIHWKIT | 1.0.0, built from source at commit `787b7e7` for sm_120, with no source changes |
| NumPy | 1.26.4 |
| tqdm | 4.67.1 |
| GPU | NVIDIA RTX 5090 |

### Building AIHWKIT for sm_120 (RTX 50 series)

```bash
conda create -n aihwkit python=3.10
conda activate aihwkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/IBM/aihwkit.git
cd aihwkit
# remove torch and torchvision from requirements.txt first
pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export USE_CUDA=ON
export TORCH_CUDA_ARCH_LIST="12.0"
export CMAKE_ARGS="-DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES=120"
rm -rf _skbuild/ build/ *.egg-info/ dist/
python setup.py build_ext --inplace -DUSE_CUDA=ON -DRPU_CUDA_ARCHITECTURES="120" --verbose
pip install -e .
```

The compiled extension needs `GLIBCXX_3.4.32`, which the conda environment's `libstdc++` does not provide. Preload the system library on activation:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Checkpoints

The checkpoints and the inference results are on Hugging Face, in [MarvinZhw/AIMC under CNN_HWA](https://huggingface.co/MarvinZhw/AIMC/tree/main/CNN_HWA). Their folders mirror `checkpoints/` and `results/` here:

```bash
hf download MarvinZhw/AIMC --include "CNN_HWA/*" --local-dir hf_aimc
cp -r hf_aimc/CNN_HWA/checkpoints hf_aimc/CNN_HWA/results .
```

| File | Contents | Results |
|---|---|---|
| `fp_cnn.th` | FP ResNet-32 | baseline `err_FP` |
| `hwa_model_final.th` | HWA model 0, seed 42 | `inference_results_<t>.csv` |
| `hwa_model_final_{1,2,3,4}.th` | HWA models 1–4, different seeds | `inference_results_<t>_{1,2,3,4}.csv` |

The Hugging Face README describes each result file.

## Usage

Run from the repository root inside the environment:

```bash
python fp_eval.py                           # FP test metrics
python hwa_train.py --seed 1 --run_id 1     # -> checkpoints/hwa_model_final_1.th (and hwa_model_1.th, best on test)
python hwa_eval.py                          # hwa_model_final.th at 1 year, 1 week, 1 day, 1 hour, 1 s
python hwa_inference.py                     # sweep of hwa_model_final_{1..4}.th -> results/inference_results_<t>_<k>.csv
```

- Without `--run_id`, `hwa_train.py` writes `checkpoints/hwa_model.th` and `hwa_model_final.th`, and overwrites model 0.
- The sweep grid is in `config/hwa_inference_config.py`:
  - 17 noise scales (0.005–2.0).
  - 3 drift scales (0.05, 0.5, 1.0).
  - 12 values of `g_min` (0–15 µS) with `g_max = 25` µS.
  - 5 inference times.
  - 25 evaluations per configuration.

  That is 612 rows per output file.
- `hwa_inference.py` skips configurations that are already in the output file, so an interrupted sweep can be resumed by running it again.

### Output columns

`results/inference_results_<t>_<k>.csv`, with `<t>` one of `second`, `hour`, `day`, `week`, `year`:

| Column | Meaning |
|---|---|
| `t_inference` | Time after programming, s |
| `noise_scale` | Multiplier of the PCM programming and read noise |
| `drift_scale` | Multiplier of the PCM drift |
| `g_min`, `g_max` | Conductance range, µS |
| `memory_window` | `g_max - g_min`, µS |
| `loss` | Test cross-entropy, nats |
| `accuracy`, `error` | Top-1 test accuracy and `1 - accuracy` |
| `norm_accuracy`, `norm_error` | Normalized accuracy A and `1 - A` |

## Legacy HPC scripts: do not use

`hpc.sh` and `hwa_inference_slurm.py` ran the model-0 sweep on Lehigh's Hawk cluster as a SLURM array, one configuration per task. That sweep used 4 drift scales: 0.005, 0.05, 0.5 and 1.0. The scripts are kept for reference only:

- They depend on the cluster: its modules, its shared conda environment and a personal account.
- `hwa_inference_slurm.py` maps job ids onto the current grid in `config/hwa_inference_config.py`, which is not the grid used for model 0.
- It also appends to `results/inference_results_<t>.csv`.

Use `hwa_inference.py` instead.

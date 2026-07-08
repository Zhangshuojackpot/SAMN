# SAMN: Self-Adaptive Monotonic Normalization for Long-Tailed Recognition

**CVPR 2026**

## Overview

SAMN addresses long-tailed recognition by constraining the classifier head's per-class weight norms to be **monotone non-decreasing** with respect to class frequency. The core insight is that in standard training, tail-class weight norms collapse relative to head classes, degrading minority-class logits. SAMN enforces a monotone ordering via a parallel GPU implementation of the **Pool Adjacent Violators Algorithm (PAVA)**, yielding a drop-in replacement (`PAVALinear`) for the standard linear head.

Training follows a standard **two-stage decoupled** protocol:
- **Stage 1** (200 epochs): train backbone + standard head jointly
- **Stage 2** (20 epochs): freeze backbone, re-train only the `PAVALinear` head

## Requirements

```bash
pip install -r requirements.txt
```

CUDA 11.8 is recommended (tested with PyTorch 2.4.1+cu118). The PAVA solver uses `torch.jit.script` and runs on GPU.

**Tested environment:**
- Python 3.12.2
- PyTorch 2.4.1 + CUDA 11.8
- torchvision 0.19.1
- numpy 1.24.4

## Data Preparation

### CIFAR-10-LT and CIFAR-100-LT

**CIFAR-10** is downloaded automatically by torchvision on first run.

**CIFAR-100** must be downloaded manually:

```bash
mkdir -p data
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz
cd ..
```

Expected layout after setup:

```
data/
├── cifar-10-batches-py/      # auto-downloaded by torchvision
│   ├── data_batch_1
│   ├── ...
│   └── test_batch
└── cifar-100-python/         # manually downloaded
    ├── train
    ├── test
    └── meta
```

> The imbalance split is generated deterministically at runtime from `--imbanlance_rate` and `--seed`. No separate split files are needed.

## Training

All experiments use **seed 3407** (set via `--seed 3407`, which is the default).

### Stage 1 — Backbone Training

Run all 12 Stage 1 experiments (CIFAR-10 and CIFAR-100, IF = 100 / 50 / 10, with and without GLMC dual-loader):

```bash
bash launch_stage1_cifar.sh
```

Checkpoints are saved to `SAMN-CVPR2026/output/<run_name>/ckpt.best.pth.tar`.

**Stage 1 hyperparameters (fixed across all CIFAR runs):**

| Parameter | Value |
|---|---|
| Architecture | ResNet-32 |
| Epochs | 200 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Weight decay | 5e-3 |
| Seed | 3407 |

`--use_glmc 1.` enables the dual-DataLoader + MixUp/CutMix + contrastive loss (GLMC backbone).  
`--use_glmc 0.` trains with standard cross-entropy (CE backbone).

### Stage 2 — Classifier Head Fine-tuning

After Stage 1 completes, run all 24 Stage 2 experiments:

```bash
bash launch_stage2_cifar.sh
```

Stage 2 fine-tunes only the classifier head for 20 epochs with the backbone frozen. Results (best epoch accuracy) are saved in `SAMN-CVPR2026/fineoutput/<run_name>/ckpt.best.pth.tar`.

**Key Stage 2 flags:**

| Flag | Value | Meaning |
|---|---|---|
| `--use_samn 1` | SAMN | `PAVALinear` head (monotone norms) |
| `--use_samn 0` | Baseline | Standard `nn.Linear` head |
| `--methods CE` | CE loss | Cross-entropy on PAVALinear |
| `--methods MISLAS` | SLAS loss | Label-Aware Smoothing |
| `--methods GLMC` | GLMC loss | Weighted sampling + low LR |

**Method-to-paper-name mapping:**

| Script flags | Paper name |
|---|---|
| `--methods MISLAS --use_samn 0` | SLAS (baseline) |
| `--methods CE --use_samn 1` | CE + SAMN |
| `--methods MISLAS --use_samn 1` | SLAS + SAMN |
| `--methods GLMC --use_samn 1` | GLMC + SAMN |

## Code Structure

```
.
├── main.py                        # Stage 1 entry point
├── Stage2.py                      # Stage 2 entry point
├── Trainer.py                     # Stage 1 training loop
├── Stage2_Trainer.py              # Stage 2 training loop
├── mislas.py                      # Label-Aware Smoothing loss (SLAS)
├── launch_stage1_cifar.sh         # Stage 1 launch script
├── launch_stage2_cifar.sh         # Stage 2 launch script
├── model/
│   ├── PAVA_Linear.py             # SAMN core: PAVALinear + parallel PAVA solver
│   ├── ResNet_cifar.py            # ResNet-32 backbone (Stage 1)
│   ├── ResNet_cifar_stage2.py     # ResNet-32 backbone (Stage 2, with PAVALinear head)
│   └── Resnet_LT.py               # ResNeXt-50 backbone (ImageNet-LT / iNaturalist)
├── imbalance_data/
│   ├── cifar10Imbanlance.py       # CIFAR-10-LT dataset with exponential imbalance
│   ├── cifar100Imbanlance.py      # CIFAR-100-LT dataset with exponential imbalance
│   └── dataset_lt_data.py         # ImageNet-LT / iNaturalist loader
├── utils/
│   ├── util.py                    # Data transforms, AverageMeter, checkpoint I/O
│   ├── moco_loader.py             # MoCo-style augmentation
│   └── randaugment.py             # RandAugment
└── data/
    └── data_txt/
        ├── ImageNet_LT_train.txt  # ImageNet-LT split (from original dataset)
        ├── ImageNet_LT_test.txt
        ├── iNaturalist18_train.txt
        └── iNaturalist18_val.txt
```

### Key files

**`model/PAVA_Linear.py`** contains the full SAMN implementation:
- `pava_parallel_fwd`: JIT-compiled O(log N) parallel PAVA solver
- `PAVAFunction`: autograd wrapper with efficient per-block gradient averaging
- `MonotonePerClassNorm`: learnable monotone scaling module
- `PAVALinear`: drop-in replacement for `nn.Linear` with monotone weight norms

## Acknowledgements

This codebase builds on [GLMC (CVPR 2023)](https://github.com/dorishuml/GLMC). The SLAS loss (`mislas.py`) is from [MiSLAS (CVPR 2021)](https://github.com/dvlab-research/MiSLAS).

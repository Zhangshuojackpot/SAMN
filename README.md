# Hyperspherical-Margin-Weighting (HMW)
This is the official PyTorch implementation of our work [Why Not Hyperparameter-Friendly Optimisation? A Monotonic Adaptive Norm Rescaling Approach For Long-Tailed Recognition](https://arxiv.org/abs/2606.02526), which has been published in CVPR2026. This repo contains some key code for our SAMN and its application to the CIFAR10/CIFAR100 dataset.<br>
<div align=center>
<img width="800" src="https://github.com/Zhangshuojackpot/SAMN/blob/main/poster.png"/>
</div>

### Abstract
Long-tailed recognition poses a significant challenge for deep learning. The two-stage decoupling paradigm, which separates representation learning from classifier retraining, offers a promising solution. During the classifier retraining stage, adaptive norm rescaling is a popular technique. It adjusts the per-class weight norms via parameter regularization, which inevitably introduces hyperparameters. However, many studies report that long-tailed recognition is sensitive to these hyperparameters, as their setup significantly impacts performance. In this paper, we first provide a class-conditional distribution perspective to support norm rescaling methods. Furthermore, we propose a simple but effective approach called Self-Adaptive Monotonic Normalization (SAMN). SAMN avoids the need for parameter regularization. It directly enforces monotonicity on per-class weight norms using the Pool Adjacent Violators Algorithm, making the method hyperparameter-friendly. SAMN is a universal strategy that integrates seamlessly with other methods for enhanced performance. Experiments on benchmark datasets demonstrate that our method significantly boosts long-tailed recognition performance, often achieving state-of-the-art results.

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


### Stage 1 — Backbone Training

Run all 12 Stage 1 experiments (CIFAR-10 and CIFAR-100, IF = 100 / 50 / 10, with and without GLMC dual-loader):

```bash
bash launch_stage1_cifar.sh
```

Checkpoints are saved to `SAMN-CVPR2026/output/<run_name>/ckpt.best.pth.tar`.

`--use_glmc 1.` enables the GLMC.  
`--use_glmc 0.` trains with standard cross-entropy (CE backbone).


### Stage 2 — Classifier Head Fine-tuning

After Stage 1 completes, run all 24 Stage 2 experiments:

```bash
bash launch_stage2_cifar.sh
```

Stage 2 fine-tunes only the classifier head for 20 epochs with the backbone frozen. Results (best epoch accuracy) are saved in `SAMN-CVPR2026/fineoutput/<run_name>/ckpt.best.pth.tar`.


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
│   └── cifar100Imbanlance.py      # CIFAR-100-LT dataset with exponential imbalance
│  
└── utils/
    ├── util.py                    # Data transforms, AverageMeter, checkpoint I/O
    ├── moco_loader.py             # MoCo-style augmentation
    └── randaugment.py             # RandAugment

```

### Key files

**`model/PAVA_Linear.py`** contains the full SAMN implementation:
- `pava_parallel_fwd`: JIT-compiled O(log N) parallel PAVA solver
- `PAVAFunction`: autograd wrapper with efficient per-block gradient averaging
- `MonotonePerClassNorm`: learnable monotone scaling module
- `PAVALinear`: drop-in replacement for `nn.Linear` with monotone weight norms

## Acknowledgements

Please consider citing GLMC and MiSLAS in your publications if it helps your research. :)

@inproceedings{zhong2021mislas,
    title={Improving Calibration for Long-Tailed Recognition},
    author={Zhisheng Zhong, Jiequan Cui, Shu Liu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021},
}

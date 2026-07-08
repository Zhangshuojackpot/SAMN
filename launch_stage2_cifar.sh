# ============================================================
# MISLAS baseline  (Stage1: CE backbone, use_glmc=0; no PAVALinear)
# ============================================================
python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --smooth_head 0.4 --smooth_tail 0.1 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 6 --smooth_head 0.3 --smooth_tail 0.0 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1  --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 4 --smooth_head 0.2 --smooth_tail 0.0 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.01 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --smooth_head 0.3 --smooth_tail 0.0 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.02 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --smooth_head 0.2 --smooth_tail 0.0 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.1  --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 2 --smooth_head 0.1 --smooth_tail 0.0 --use_samn 0 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'


# ============================================================
# SAMN + CE  (Stage1: CE backbone, use_glmc=0; PAVALinear + CE loss)
# ============================================================
python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --epochs 20 -b 64 --methods CE --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --epochs 20 -b 64 --methods CE --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 6 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1  --epochs 20 -b 64 --methods CE --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 4 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.01 --epochs 20 -b 64 --methods CE --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.02 --epochs 20 -b 64 --methods CE --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.1  --epochs 20 -b 64 --methods CE --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 2 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'


# ============================================================
# SAMN + MISLAS  (Stage1: CE backbone, use_glmc=0; PAVALinear + MISLAS loss)
# ============================================================
python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --smooth_head 0.01 --smooth_tail 0.0 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 6 --smooth_head 0.01 --smooth_tail 0.0 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1  --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 4 --smooth_head 0.01 --smooth_tail 0.0 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.01 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --smooth_head 0.3 --smooth_tail 0.1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.01#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.02 --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --smooth_head 0.3 --smooth_tail 0.1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.02#use_glmc:0.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.1  --epochs 20 -b 64 --methods MISLAS --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 2 --smooth_head 0.3 --smooth_tail 0.1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.1#use_glmc:0.0/ckpt.best.pth.tar'


# ============================================================
# SAMN + GLMC  (Stage1: GLMC backbone, use_glmc=1; PAVALinear + GLMC)
# ============================================================
python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --epochs 20 -b 64 --methods GLMC --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.01#use_glmc:1.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --epochs 20 -b 64 --methods GLMC --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 6 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.02#use_glmc:1.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1  --epochs 20 -b 64 --methods GLMC --resample_weighting 0.2  --label_weighting 1.2 --contrast_weight 4 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar100#arch: resnet32#imbanlance_rate: 0.1#use_glmc:1.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.01 --epochs 20 -b 64 --methods GLMC --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.01#use_glmc:1.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.02 --epochs 20 -b 64 --methods GLMC --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.02#use_glmc:1.0/ckpt.best.pth.tar'

python Stage2.py --dataset cifar10  -a resnet32 --num_classes 10  --imbanlance_rate 0.1  --epochs 20 -b 64 --methods GLMC --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 2 --use_samn 1 --resume './SAMN-CVPR2026/output/dataset: cifar10#arch: resnet32#imbanlance_rate: 0.1#use_glmc:1.0/ckpt.best.pth.tar'

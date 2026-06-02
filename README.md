# Self-Adaptive Monotonic Normalization (SAMN)
This is the official PyTorch implementation of our work [Why Not Hyperparameter-Friendly Optimisation? A Monotonic Adaptive Norm
Rescaling Approach For Long-Tailed Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Why_Not_Hyperparameter-Friendly_Optimisation_A_Monotonic_Adaptive_Norm_Rescaling_Approach_CVPR_2026_paper.html), which has been published in CVPR-26. This repo contains some key codes of our SAMN and its application in CIFAR10/CIFAR100 dataset.<br>

<div align=center>
<img width="900" src="https://github.com/Zhangshuojackpot/SAMN/blob/main/poster.png"/>
</div>

### Abstract
Long-tailed recognition poses a significant challenge for deep learning. The two-stage decoupling paradigm, which separates representation learning from classifier retraining, offers a promising solution. During the classifier retraining stage, adaptive norm rescaling is a popular technique. It adjusts the per-class weight norms via parameter regularization, which inevitably introduces hyperparameters. However, many studies report that long-tailed recognition is sensitive to these hyperparameters, as their setup significantly impacts performance. In this paper, we first provide a class-conditional distribution perspective to support norm rescaling methods. Furthermore, we propose a simple but effective approach called Self-Adaptive Monotonic Normalization (SAMN). SAMN avoids the need for parameter regularization. It directly enforces monotonicity on per-class weight norms using the Pool Adjacent Violators Algorithm, making the method hyperparameter-friendly. SAMN is a universal strategy that integrates seamlessly with other methods for enhanced performance. Experiments on benchmark datasets demonstrate that our method significantly boosts long-tailed recognition performance, often achieving state-of-the-art results.

### Codes are coming soon!


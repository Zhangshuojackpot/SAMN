import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Sequence, Optional, Tuple


# -----------------------------------------------------------------------------
# Parallel PAVA solver (JIT): O(log N) iterations vs. serial O(N) loop
# -----------------------------------------------------------------------------
@torch.jit.script
def pava_parallel_fwd(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pool Adjacent Violators Algorithm — parallel GPU implementation.
    Returns (unique_values, counts): compressed monotone sequence and block sizes.
    """
    device = y.device
    v = y.clone()
    w = torch.ones_like(y)

    while True:
        if v.numel() == 1:
            break

        boundary = v[:-1] <= v[1:]
        if boundary.all():
            break

        # Prepend True so element 0 always starts a new block
        pad = torch.ones(1, dtype=torch.bool, device=device)
        boundary_full = torch.cat((pad, boundary))

        # cumsum produces group numbers: [0, 0, 1, 2, 2, ...]
        group_ids = torch.cumsum(boundary_full.long(), dim=0) - 1
        num_groups = group_ids[-1].item() + 1

        # num_groups is int; cast explicitly for JIT safety
        new_w = torch.zeros(int(num_groups), dtype=v.dtype, device=device)
        new_vw = torch.zeros(int(num_groups), dtype=v.dtype, device=device)

        new_w.index_add_(0, group_ids, w)
        new_vw.index_add_(0, group_ids, v * w)

        v = new_vw / new_w
        w = new_w

    return v, w


# -----------------------------------------------------------------------------
# Autograd Function: efficient backward via per-block gradient averaging
# -----------------------------------------------------------------------------
class PAVAFunction(Function):
    @staticmethod
    def forward(ctx, y):
        with torch.no_grad():
            v_sorted, counts = pava_parallel_fwd(y)
            z = torch.repeat_interleave(v_sorted, counts.long())

        ctx.save_for_backward(counts)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        """Average gradients within each merged PAVA block."""
        counts = ctx.saved_tensors[0]
        device = grad_output.device

        # counts: [2, 1] → block_indices: [0, 0, 1]
        block_indices = torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            counts.long()
        )

        grad_sums = torch.zeros(len(counts), dtype=grad_output.dtype, device=device)
        grad_sums.index_add_(0, block_indices, grad_output)

        grad_means = grad_sums / counts
        grad_input = grad_means[block_indices]

        return grad_input


# -----------------------------------------------------------------------------
# Module definitions
# -----------------------------------------------------------------------------
class MonotonePerClassNorm(nn.Module):
    def __init__(self,
                 class_counts: Sequence[float],
                 eps: float = 1e-6,
                 strict: bool = False,
                 strict_eps: float = 1e-4,
                 init: str = "freq",
                 device: Optional[torch.device] = None):
        super().__init__()
        if isinstance(class_counts, torch.Tensor):
            class_counts = class_counts.detach().to(device)
        else:
            class_counts = torch.tensor(list(class_counts), dtype=torch.float32, device=device)
        self.register_buffer("class_counts", class_counts)
        self.C = class_counts.numel()
        self.eps = float(eps)
        self.strict = bool(strict)
        self.strict_eps = float(strict_eps)

        if init == "freq":
            minc = float(torch.min(class_counts))
            maxc = float(torch.max(class_counts))
            if maxc - minc < 1e-12:
                normal_freq = torch.zeros_like(class_counts)
            else:
                normal_freq = (class_counts - minc) / (maxc - minc)
            init_raw = normal_freq + 1e-6
        else:
            init_raw = torch.zeros_like(class_counts)

        self.raw = nn.Parameter(init_raw.clone().detach())

        # Pre-compute sort permutation to avoid re-sorting on every forward
        idx = torch.arange(self.C, device=class_counts.device, dtype=class_counts.dtype)
        keys = class_counts + idx * 1e-8
        _, perm = torch.sort(keys, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(self.C, device=class_counts.device, dtype=perm.dtype)

        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, logits: Optional[torch.Tensor] = None, *, return_temps_only: bool = False,
                return_scaling: bool = True):
        raw_sorted = self.raw[self.perm]
        proj_sorted = PAVAFunction.apply(raw_sorted)

        if self.strict:
            ranks = torch.arange(self.C, dtype=proj_sorted.dtype, device=proj_sorted.device)
            proj_sorted = proj_sorted + self.strict_eps * ranks

        temps_sorted = F.softplus(proj_sorted) + self.eps
        temps = temps_sorted[self.inv_perm]

        if return_temps_only:
            return temps

        scaling = 1.0 / (temps + self.eps)

        if logits is None:
            return scaling if return_scaling else temps
        else:
            return logits * scaling.unsqueeze(0)


class PAVALinear(nn.Module):
    """Drop-in replacement for nn.Linear with monotone per-class norm scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 class_counts: Sequence[float] = None, *, scale_mode: str = "exp", args: Optional[dict] = None):
        super().__init__()
        assert class_counts is not None, "class_counts must be provided"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.fix_norms_w = MonotonePerClassNorm(class_counts=class_counts)
        self.fix_norms_b = MonotonePerClassNorm(class_counts=class_counts)

        assert scale_mode in ("exp", "mul")
        self.scale_mode = scale_mode
        self.args = args or {}
        self.reset_parameters()

    @staticmethod
    def from_linear(linear_module: nn.Linear, class_counts: Sequence[float], *, scale_mode: str = "exp"):
        in_f = linear_module.in_features
        out_f = linear_module.out_features
        bias = linear_module.bias is not None
        new = PAVALinear(in_f, out_f, bias=bias, class_counts=class_counts, scale_mode=scale_mode)
        with torch.no_grad():
            new.weight.copy_(linear_module.weight)
            if bias and linear_module.bias is not None:
                new.bias.copy_(linear_module.bias)
        return new

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor):
        directions = F.normalize(self.weight, p=2, dim=1)
        per_class_scaling_w = self.fix_norms_w()
        per_class_scaling_b = self.fix_norms_b()
        if self.scale_mode == "exp":
            row_scales = torch.exp(per_class_scaling_w).unsqueeze(1)
            scales_b = torch.exp(per_class_scaling_b)
        else:
            row_scales = per_class_scaling_w.clamp_min(1e-6).unsqueeze(1)
            scales_b = per_class_scaling_b.clamp_min(1e-6)

        real_weight = directions * row_scales
        self.real_bias = scales_b + self.bias
        return F.linear(x, real_weight, self.real_bias)

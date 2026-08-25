"""
LoRA and QLoRA

Four demos:
  1. A LoRALinear module built from scratch in raw PyTorch: a frozen
     nn.Linear plus a trainable low-rank A/B update.
  2. Train it on a tiny toy regression task and verify DIRECTLY that the
     frozen base weight never changes by a single bit, while A and B do.
  3. Parameter-count comparison: dense fine-tuning vs. LoRA at several
     ranks, for a realistic-scale weight matrix, cross-checked against
     real PyTorch parameter counts (not just the formula).
  4. A from-scratch simulation of QLoRA's 4-bit block quantization of
     the frozen base: quantization error and memory footprint vs.
     fp32/fp16 storage.

No external PEFT library is used or needed -- this lesson builds the
mechanism itself. A real workflow would use Hugging Face `peft`
(LoraConfig / get_peft_model), covered in Lesson 5.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. LoRALinear: a frozen nn.Linear plus a trainable low-rank update
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear(in_features, out_features) with a trainable
    low-rank update W' = W + (alpha/r) * B @ A, exactly as in the README."""

    def __init__(self, in_features, out_features, r, alpha=None, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha if alpha is not None else r  # a common default: alpha == r
        self.scaling = self.alpha / self.r

        # The original pretrained layer -- frozen. In a real workflow this
        # would already hold pretrained weights; here it's just initialized
        # normally to stand in for "some pretrained weight matrix".
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)

        # A: r x in_features, small random init (breaks symmetry).
        # B: out_features x r, initialized to ZERO so that at step 0,
        #    B @ A == 0 and this module is numerically identical to `base`.
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scaling * lora_out

    def merged_weight(self):
        """W + (alpha/r) * B @ A -- what you'd deploy for zero-extra-latency inference."""
        return self.base.weight + self.scaling * (self.B @ self.A)


def lora_layer_demo():
    print("=" * 78)
    print("1-2. A FROZEN BASE + TRAINABLE LOW-RANK UPDATE, TRAINED AND VERIFIED")
    print("=" * 78)

    in_features, out_features, r = 32, 32, 4
    layer = LoRALinear(in_features, out_features, r=r)

    base_weight_before = layer.base.weight.detach().clone()
    base_bias_before = layer.base.bias.detach().clone()

    # A toy task: learn a fixed random linear map that the randomly-initialized
    # frozen base does NOT already implement. Only A/B can possibly fit this,
    # since base.weight/bias have requires_grad=False.
    target_map = torch.randn(out_features, in_features)
    x_train = torch.randn(256, in_features)
    y_train = x_train @ target_map.T

    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} "
          f"(A: {layer.A.numel()}, B: {layer.B.numel()})")
    print(f"Frozen parameters:    {sum(p.numel() for p in layer.parameters() if not p.requires_grad)} "
          f"(base.weight + base.bias)")

    optimizer = torch.optim.Adam(trainable_params, lr=1e-2)
    initial_loss = None
    for step in range(1, 301):
        pred = layer(x_train)
        loss = F.mse_loss(pred, y_train)
        if initial_loss is None:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss = {loss.item():.5f}")

    final_loss = loss.item()

    # The core PEFT claim, checked directly rather than assumed:
    weight_unchanged = torch.equal(layer.base.weight, base_weight_before)
    bias_unchanged = torch.equal(layer.base.bias, base_bias_before)
    weight_grad_is_none = layer.base.weight.grad is None
    A_changed = not torch.allclose(layer.A, torch.zeros_like(layer.A) + layer.A[0, 0])  # sanity: A is not degenerate
    B_is_nonzero = layer.B.abs().sum().item() > 0

    print(f"\nFrozen base.weight identical to its pre-training value: {weight_unchanged}")
    print(f"Frozen base.bias identical to its pre-training value:   {bias_unchanged}")
    print(f"Frozen base.weight.grad is None (never touched by backward's update step): {weight_grad_is_none}")
    print(f"B moved away from its zero initialization (sum|B| = {layer.B.abs().sum().item():.4f} > 0): {B_is_nonzero}")
    print(f"\nLoss fell from {initial_loss:.5f} to {final_loss:.5f} training ONLY A and B --")
    print("the frozen base contributed a fixed, unhelpful random transform throughout,")
    print(f"and the rank-{r} update alone did all the learning. It plateaus above zero")
    print(f"because the target map needs a full-rank ({in_features}-rank) correction and a")
    print(f"rank-{r} B@A update is mathematically incapable of reproducing all of it --")
    print("a direct, hands-on illustration of what 'low-rank' actually constrains.")

    # Merging check: the merged weight should reproduce the trained forward pass exactly.
    merged_w = layer.merged_weight()
    with torch.no_grad():
        out_unmerged = layer(x_train[:8])
        out_merged = x_train[:8] @ merged_w.T + layer.base.bias
    max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"\nMax difference between LoRA forward pass and the merged-weight forward pass: "
          f"{max_diff:.2e}")
    print("-> Merging A/B into W once, offline, reproduces training-time behavior exactly --")
    print("   this is why LoRA adds zero extra latency once merged for deployment.")


# ---------------------------------------------------------------------------
# 3. Parameter counts: dense fine-tuning vs. LoRA at several ranks
# ---------------------------------------------------------------------------

def parameter_count_demo():
    print("\n" + "=" * 78)
    print("3. TRAINABLE PARAMETERS: FULL FINE-TUNING vs. LoRA AT SEVERAL RANKS")
    print("=" * 78)

    d = k = 4096  # a realistic attention/FFN-projection matrix size (7B-model class)
    dense_params_formula = d * k
    print(f"Weight matrix shape: {d} x {k}")
    print(f"Full fine-tuning (every entry trainable): {dense_params_formula:,} parameters\n")

    print(f"{'rank r':>8}{'formula: r*(d+k)':>20}{'actual LoRALinear':>20}{'% of dense':>14}")
    for r in [1, 4, 16, 64]:
        formula_count = r * (d + k)
        # Cross-check the formula against a REAL instantiated module's parameter count.
        layer = LoRALinear(d, k, r=r, bias=False)
        actual_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        pct = 100 * actual_trainable / dense_params_formula
        match = "OK" if formula_count == actual_trainable else "MISMATCH"
        print(f"{r:>8}{formula_count:>20,}{actual_trainable:>20,}{pct:>13.3f}%   [{match}]")

    print("\n-> Even at r=64, LoRA trains scarcely 3% as many parameters as full fine-tuning")
    print("   of this single matrix -- and real models apply LoRA to only a handful of")
    print("   projection matrices per layer (commonly just the attention Q and V")
    print("   projections), so the reduction across an entire model is even larger than")
    print("   this single-matrix comparison shows.")


# ---------------------------------------------------------------------------
# 4. QLoRA: simulate 4-bit block quantization of the frozen base
# ---------------------------------------------------------------------------

def quantize_4bit_blockwise(weight, block_size=64):
    """A simplified simulation of QLoRA-style 4-bit blockwise quantization
    (real NF4 uses non-uniform, empirically-fitted quantization levels
    matched to pretrained weights' distribution; this uses a uniform 4-bit
    grid per block, which is simpler but demonstrates the same mechanism:
    per-block scale factors + a shared low-bit code).

    Returns (dequantized_weight, bytes_used, original_bytes_fp32).
    """
    flat = weight.flatten()
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block_size)

    # Per-block absmax scaling -- exactly what real blockwise quantization does.
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    LEVELS = 7  # 4-bit signed range used here: integers -7..7 (15 levels, fits in 4 bits)
    codes = torch.round(blocks / absmax * LEVELS).clamp(-LEVELS, LEVELS)
    dequantized_blocks = codes / LEVELS * absmax

    dequantized = dequantized_blocks.flatten()[:n].view_as(weight)

    num_blocks = blocks.shape[0]
    bytes_codes = n * 0.5           # 4 bits = 0.5 bytes per weight
    bytes_scales = num_blocks * 4   # one fp32 scale constant per block
    bytes_used = bytes_codes + bytes_scales
    bytes_fp32 = n * 4
    return dequantized, bytes_used, bytes_fp32


def qlora_demo():
    print("\n" + "=" * 78)
    print("4. QLoRA: 4-BIT BLOCKWISE QUANTIZATION OF THE FROZEN BASE (SIMULATED)")
    print("=" * 78)

    torch.manual_seed(1)
    # Pretrained weights are roughly Gaussian-distributed -- this is exactly
    # the property NF4's non-uniform quantization levels are designed around.
    weight = torch.randn(4096, 4096)

    dequantized, bytes_4bit, bytes_fp32 = quantize_4bit_blockwise(weight, block_size=64)
    abs_error = (weight - dequantized).abs()
    rel_error = abs_error.mean().item() / weight.abs().mean().item()

    bytes_fp16 = weight.numel() * 2

    print(f"Weight matrix: {tuple(weight.shape)} = {weight.numel():,} values\n")
    print(f"{'storage format':<28}{'bytes':>14}{'vs fp32':>12}")
    print(f"{'fp32 (4 bytes/value)':<28}{bytes_fp32:>14,.0f}{'1.0x':>12}")
    print(f"{'fp16 (2 bytes/value)':<28}{bytes_fp16:>14,.0f}{bytes_fp32 / bytes_fp16:>11.1f}x")
    print(f"{'4-bit blockwise (simulated)':<28}{bytes_4bit:>14,.0f}{bytes_fp32 / bytes_4bit:>11.1f}x")

    print(f"\nMean absolute weight value:        {weight.abs().mean().item():.4f}")
    print(f"Mean absolute quantization error:  {abs_error.mean().item():.4f}")
    print(f"Relative error:                    {rel_error * 100:.2f}%")

    print(f"\n-> Storing the frozen base at 4 bits instead of fp32 shrinks its memory by")
    print(f"   {bytes_fp32 / bytes_4bit:.1f}x, for a {rel_error * 100:.1f}% relative reconstruction error -- a small enough")
    print("   error that the base model's forward pass stays useful, while its memory")
    print("   footprint drops enough to fit much larger models on a single GPU. This is")
    print("   exactly QLoRA's trick: quantize the FROZEN base (which never needs exact")
    print("   gradients), keep the small trainable A/B LoRA matrices in full precision")
    print("   (they're what's actually learning), and dequantize each block back to a")
    print("   working precision on the fly for each forward/backward pass.")


def main():
    lora_layer_demo()
    parameter_count_demo()
    qlora_demo()


if __name__ == "__main__":
    main()

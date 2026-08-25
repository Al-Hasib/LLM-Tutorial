"""
Layer Norm, Residuals and Feed-Forward Sublayers

Three demos:
  1. Residual connections keeping gradients alive through a very deep
     stack, measured directly with PyTorch autograd (compare with vs.
     without the "+x" skip connection).
  2. A hand-rolled LayerNorm, verified numerically against nn.LayerNorm.
  3. Parameter counts: attention vs. feed-forward sublayer, confirming
     the "FFN holds ~2/3 of a layer's parameters" claim from the README.

Run:
    python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Residual connections keep gradients alive through depth
# ---------------------------------------------------------------------------

class SimpleBlock(nn.Module):
    """A minimal stand-in sublayer: Linear -> Tanh -> Linear, small init."""

    def __init__(self, d, seed_offset=0):
        super().__init__()
        torch.manual_seed(seed_offset)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        with torch.no_grad():
            self.fc1.weight *= 0.5
            self.fc2.weight *= 0.5

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class DeepStack(nn.Module):
    def __init__(self, d, depth, use_residual):
        super().__init__()
        self.blocks = nn.ModuleList([SimpleBlock(d, seed_offset=i) for i in range(depth)])
        self.use_residual = use_residual

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x) if self.use_residual else block(x)
        return x


def residual_gradient_demo():
    print("=" * 70)
    print("1. RESIDUAL CONNECTIONS AND GRADIENT FLOW THROUGH DEPTH")
    print("=" * 70)
    print("Measuring the gradient norm that reaches the INPUT of a deep stack,")
    print("with vs. without residual ('x = x + block(x)' vs. 'x = block(x)').\n")

    d = 16
    print(f"{'depth':>6}  {'grad norm, NO residual':>24}  {'grad norm, WITH residual':>26}")
    for depth in [5, 10, 20, 40, 80]:
        x_no_res = torch.randn(d, requires_grad=True)
        stack_no_res = DeepStack(d, depth, use_residual=False)
        out_no_res = stack_no_res(x_no_res).sum()
        out_no_res.backward()
        grad_norm_no_res = x_no_res.grad.norm().item()

        x_res = torch.randn(d, requires_grad=True)
        stack_res = DeepStack(d, depth, use_residual=True)
        out_res = stack_res(x_res).sum()
        out_res.backward()
        grad_norm_res = x_res.grad.norm().item()

        print(f"{depth:>6}  {grad_norm_no_res:>24.8f}  {grad_norm_res:>26.8f}")

    print("\n-> Without residual connections, the gradient reaching the input")
    print("   shrinks toward zero as depth grows -- the same vanishing-signal")
    print("   story as Phase 01's RNN gradient decay, just caused by stacking")
    print("   layers instead of stacking timesteps. With residual connections,")
    print("   the identity path (+x) guarantees a gradient of at least 1 flows")
    print("   straight back regardless of depth, keeping the total gradient")
    print("   from collapsing.")


# ---------------------------------------------------------------------------
# 2. Hand-rolled LayerNorm vs. nn.LayerNorm
# ---------------------------------------------------------------------------

def manual_layer_norm(x, gamma, beta, eps=1e-5):
    """x: (..., d). Normalize over the LAST dimension only (per-token, not per-batch)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta


def layer_norm_demo():
    print("\n" + "=" * 70)
    print("2. HAND-ROLLED LayerNorm vs. torch.nn.LayerNorm")
    print("=" * 70)

    batch, T, d_model = 2, 4, 8
    x = torch.randn(batch, T, d_model) * 5 + 3   # deliberately not mean-0/std-1

    torch_ln = nn.LayerNorm(d_model)
    with torch.no_grad():
        # Use the same (randomly initialized) gamma/beta for both, for a fair comparison.
        gamma, beta = torch_ln.weight.clone(), torch_ln.bias.clone()

    manual_output = manual_layer_norm(x, gamma, beta)
    torch_output = torch_ln(x)

    max_diff = (manual_output - torch_output).abs().max().item()
    print(f"Input stats:  mean={x.mean().item():.3f}  std={x.std().item():.3f}")
    print(f"Output stats (torch.nn.LayerNorm), per-token: mean~0, std~1 before "
          f"gamma/beta scaling")
    print(f"Max abs difference between manual and torch.nn.LayerNorm: {max_diff:.2e}")

    per_token_mean = torch_output.mean(dim=-1)
    print(f"\nPer-token means after LayerNorm: {per_token_mean[0].detach().numpy().round(3)}")
    print("(Exactly 0 here only because nn.LayerNorm's beta defaults to all-zeros and")
    print(" gamma to all-ones; with a trained/non-default beta the mean would shift to")
    print(" beta's own mean, since normalization guarantees mean 0 BEFORE gamma/beta.)")


# ---------------------------------------------------------------------------
# 3. Parameter counts: attention vs. feed-forward
# ---------------------------------------------------------------------------

class MultiHeadAttentionParams(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


class FeedForwardParams(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def parameter_count_demo():
    print("\n" + "=" * 70)
    print("3. PARAMETER COUNTS: ATTENTION vs. FEED-FORWARD SUBLAYER")
    print("=" * 70)

    d_model = 768   # GPT-2-small scale, for a realistic comparison
    d_ff = 4 * d_model

    attn = MultiHeadAttentionParams(d_model)
    ffn = FeedForwardParams(d_model, d_ff)

    attn_params = count_params(attn)
    ffn_params = count_params(ffn)
    total = attn_params + ffn_params

    print(f"d_model={d_model}, d_ff={d_ff} (the conventional 4x d_model)")
    print(f"  Attention sublayer params: {attn_params:,}")
    print(f"  Feed-forward sublayer params: {ffn_params:,}")
    print(f"  FFN's share of one layer's total: {ffn_params / total:.1%}")
    print("\n-> Matches the README's claim: the feed-forward sublayer holds")
    print("   roughly two-thirds of a Transformer layer's parameters, even")
    print("   though self-attention gets most of the conceptual spotlight.")


def main():
    residual_gradient_demo()
    layer_norm_demo()
    parameter_count_demo()


if __name__ == "__main__":
    main()

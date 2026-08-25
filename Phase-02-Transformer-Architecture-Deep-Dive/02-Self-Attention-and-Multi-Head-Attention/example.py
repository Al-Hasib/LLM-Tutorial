"""
Self-Attention and Multi-Head Attention

PyTorch implementations of scaled dot-product attention and multi-head
attention, plus two concrete demos:
  1. Why the 1/sqrt(d_k) scaling factor matters -- softmax saturation
     with and without it, as d_k grows.
  2. Causal masking -- verifying a decoder position truly cannot see
     future positions.

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Why scale by sqrt(d_k)? Measure softmax saturation directly.
# ---------------------------------------------------------------------------

def scaling_demo():
    print("=" * 70)
    print("1. WHY SCALE BY 1/sqrt(d_k)?")
    print("=" * 70)
    print("Same random Q, K distributions, varying d_k. We look at the max")
    print("softmax weight in a row -- close to 1.0 means the distribution has")
    print("collapsed onto a single key (saturation), which starves gradients")
    print("everywhere else.\n")

    T = 10  # number of keys being attended over
    print(f"{'d_k':>6}  {'raw score std':>14}  {'max weight (unscaled)':>23}  "
          f"{'max weight (scaled)':>21}")
    for d_k in [4, 16, 64, 256, 1024]:
        q = torch.randn(d_k)
        K = torch.randn(T, d_k)
        raw_scores = K @ q                              # (T,) unscaled dot products
        scaled_scores = raw_scores / math.sqrt(d_k)       # (T,) scaled

        unscaled_weights = F.softmax(raw_scores, dim=-1)
        scaled_weights = F.softmax(scaled_scores, dim=-1)

        print(f"{d_k:>6}  {raw_scores.std().item():>14.2f}  "
              f"{unscaled_weights.max().item():>23.4f}  "
              f"{scaled_weights.max().item():>21.4f}")

    print("\n-> As d_k grows, unscaled scores' spread grows too (variance scales")
    print("   with d_k, as derived in the README), and the unscaled softmax")
    print("   collapses toward a one-hot distribution (max weight -> 1.0).")
    print("   The scaled version stays well-behaved regardless of d_k.")


# ---------------------------------------------------------------------------
# 2. Scaled dot-product attention (the core building block)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (..., T, d_k) / (..., T, d_v). mask: (..., T, T), True/1 = keep.
    Returns (output, attention_weights)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


# ---------------------------------------------------------------------------
# 3. Multi-head attention as a reusable nn.Module
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (batch, T, d_model) -> (batch, num_heads, T, d_k)
        batch, T, d_model = x.shape
        x = x.view(batch, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        # (batch, num_heads, T, d_k) -> (batch, T, d_model)
        batch, num_heads, T, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, T, num_heads * d_k)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))   # (batch, heads, T, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        combined = self.combine_heads(attn_output)      # (batch, T, d_model)
        output = self.W_o(combined)
        return output, attn_weights


def causal_mask(T):
    """Lower-triangular mask: position i may attend to positions <= i only."""
    return torch.tril(torch.ones(T, T)).bool()


def multi_head_demo():
    print("\n" + "=" * 70)
    print("2. MULTI-HEAD ATTENTION (PyTorch nn.Module)")
    print("=" * 70)

    batch, T, d_model, num_heads = 1, 6, 16, 4
    x = torch.randn(batch, T, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output, attn_weights = mha(x)   # no mask -> full (bidirectional) self-attention

    print(f"input shape:  {tuple(x.shape)}  (batch, seq_len, d_model)")
    print(f"output shape: {tuple(output.shape)}  (unchanged -- attention preserves shape)")
    print(f"attn_weights shape: {tuple(attn_weights.shape)}  (batch, heads, T, T)")
    print(f"num_heads={num_heads}, d_k per head={mha.d_k}  "
          f"(heads x d_k = {num_heads * mha.d_k} = d_model, same total budget)")

    print("\nEach row of each head's attention matrix sums to 1 (verifying softmax):")
    for h in range(num_heads):
        row_sums = attn_weights[0, h].sum(dim=-1)
        print(f"  head {h}: row sums = {[round(s, 4) for s in row_sums.tolist()]}")


def causal_masking_demo():
    print("\n" + "=" * 70)
    print("3. CAUSAL MASKING: CAN A DECODER POSITION SEE THE FUTURE?")
    print("=" * 70)

    batch, T, d_model, num_heads = 1, 5, 16, 2
    x = torch.randn(batch, T, d_model)
    mask = causal_mask(T)
    print("Causal mask (True = allowed to attend):")
    print(mask.int())

    mha = MultiHeadAttention(d_model, num_heads)
    _, attn_weights = mha(x, mask=mask)

    head0_weights = attn_weights[0, 0]
    print("\nHead 0's attention weight matrix (rows = query position,")
    print("columns = key position), rounded to 3 decimals:")
    print(head0_weights.detach().numpy().round(3))

    upper_triangular_mass = 0.0
    for i in range(T):
        upper_triangular_mass += head0_weights[i, i + 1:].sum().item()
    print(f"\nTotal attention weight placed on FUTURE positions: {upper_triangular_mass:.10f}")
    print("-> Exactly 0.0 (up to floating point): masking guarantees a decoder")
    print("   position can never attend to a position that comes after it.")


def main():
    scaling_demo()
    multi_head_demo()
    causal_masking_demo()


if __name__ == "__main__":
    main()

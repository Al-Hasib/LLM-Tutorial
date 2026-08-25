"""
Survey of Popular Open LLMs

Two things:
  1. A working Grouped-Query Attention (GQA) implementation in PyTorch,
     generalizing Phase 02's MultiHeadAttention -- set num_kv_heads ==
     num_heads for ordinary MHA, or num_kv_heads == 1 for MQA.
  2. A direct KV-cache-size comparison across MHA / GQA / MQA at a
     realistic model scale, and a static architecture-comparison table
     across the surveyed model families.

Run:
    python example.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Grouped-Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """num_kv_heads == num_heads  -> ordinary multi-head attention (Phase 02).
    num_kv_heads == 1             -> multi-query attention.
    otherwise                     -> grouped-query attention (the general case)."""

    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads   # how many Q heads share one K/V head
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)     # FEWER params than MHA's W_k
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model)

    def forward(self, x, mask=None):
        batch, T, _ = x.shape

        Q = self.W_q(x).view(batch, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Each K/V head is REUSED by `group_size` consecutive Q heads.
        K = K.repeat_interleave(self.group_size, dim=1)   # (batch, num_heads, T, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(batch, T, self.num_heads * self.d_k)
        return self.W_o(out)


def gqa_forward_demo():
    print("=" * 70)
    print("1. GROUPED-QUERY ATTENTION: ONE MODULE, THREE VARIANTS")
    print("=" * 70)

    d_model, num_heads, T, batch = 32, 8, 5, 1
    x = torch.randn(batch, T, d_model)

    for name, num_kv_heads in [("MHA (num_kv_heads=8)", 8),
                                ("GQA (num_kv_heads=4)", 4),
                                ("GQA (num_kv_heads=2)", 2),
                                ("MQA (num_kv_heads=1)", 1)]:
        attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        output = attn(x)
        kv_params = sum(p.numel() for p in [attn.W_k.weight, attn.W_v.weight])
        print(f"  {name:24s} output shape {tuple(output.shape)}   "
              f"K+V projection params: {kv_params}")

    print("\n-> Output shape is identical across all three -- GQA/MQA are drop-in")
    print("   replacements for MHA. Only the K/V projection size (and therefore")
    print("   the KV cache built from it) shrinks.")


# ---------------------------------------------------------------------------
# 2. KV-cache size: MHA vs. GQA vs. MQA, at a realistic scale
# ---------------------------------------------------------------------------

def kv_cache_bytes(batch, seq_len, num_kv_heads, d_k, num_layers, bytes_per_value=2):
    """2x for storing both K and V, one cache entry per layer."""
    return 2 * batch * seq_len * num_kv_heads * d_k * num_layers * bytes_per_value


def kv_cache_demo():
    print("\n" + "=" * 70)
    print("2. KV-CACHE SIZE AT A REALISTIC SCALE (LLaMA-2-70B-like config)")
    print("=" * 70)

    batch, seq_len = 1, 4096
    num_heads, d_k, num_layers = 64, 128, 80   # d_model = 8192
    print(f"batch={batch}, seq_len={seq_len}, num_heads={num_heads}, d_k={d_k}, "
          f"num_layers={num_layers}, fp16 (2 bytes/value)\n")

    print(f"{'variant':>22}{'num_kv_heads':>15}{'KV cache size':>18}")
    for name, num_kv_heads in [("MHA", 64), ("GQA (8 groups)", 8), ("MQA", 1)]:
        cache_bytes = kv_cache_bytes(batch, seq_len, num_kv_heads, d_k, num_layers)
        cache_gb = cache_bytes / (1024 ** 3)
        print(f"{name:>22}{num_kv_heads:>15}{cache_gb:>15.2f} GB")

    print("\n-> Going from MHA to 8-group GQA shrinks the KV cache by 8x with only")
    print("   8 distinct K/V projections instead of 64 -- this is exactly why")
    print("   LLaMA 2 70B and LLaMA 3 use GQA: it makes serving long conversations")
    print("   to many users at once dramatically cheaper in accelerator memory,")
    print("   at a much smaller quality cost than going all the way to MQA.")


# ---------------------------------------------------------------------------
# 3. A static architecture-comparison table across the surveyed models
# ---------------------------------------------------------------------------

MODEL_SURVEY = [
    # name, position scheme, normalization, activation, attention variant, MoE?
    ("LLaMA 2 7B",   "RoPE",  "RMSNorm", "SwiGLU", "MHA", "no"),
    ("LLaMA 2 70B",  "RoPE",  "RMSNorm", "SwiGLU", "GQA", "no"),
    ("Mistral 7B",   "RoPE",  "RMSNorm", "SwiGLU", "GQA + sliding-window", "no"),
    ("Mixtral 8x7B", "RoPE",  "RMSNorm", "SwiGLU", "GQA + sliding-window", "yes (8 experts, top-2)"),
    ("GPT-2",        "learned", "LayerNorm", "GELU", "MHA", "no"),
]


def architecture_survey_demo():
    print("\n" + "=" * 70)
    print("3. ARCHITECTURE COMPARISON ACROSS THE SURVEYED MODELS")
    print("=" * 70)
    header = f"{'model':<16}{'position':<10}{'norm':<12}{'activation':<12}" \
             f"{'attention':<24}{'MoE':<24}"
    print(header)
    for name, pos, norm, act, attn, moe in MODEL_SURVEY:
        print(f"{name:<16}{pos:<10}{norm:<12}{act:<12}{attn:<24}{moe:<24}")

    print("\n-> Read across a row and every entry maps to a specific lesson in this")
    print("   course: position -> Phase 02 L3 / Phase 03 L6, norm/activation ->")
    print("   Phase 02 L5, attention variant -> Phase 02 L2 / this lesson's GQA /")
    print("   Phase 03 L6's sliding window, MoE -> Phase 03 L4. No row introduces")
    print("   a concept this course hasn't already built from scratch.")


def main():
    gqa_forward_demo()
    kv_cache_demo()
    architecture_survey_demo()


if __name__ == "__main__":
    main()

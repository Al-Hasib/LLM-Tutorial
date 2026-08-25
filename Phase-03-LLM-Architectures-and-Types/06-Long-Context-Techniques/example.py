"""
Long-Context Techniques

Three demos:
  1. RoPE -- rotating Q and K by position, verifying the dot product
     depends ONLY on relative position, never absolute position.
  2. ALiBi -- a parameter-free distance penalty added directly to
     attention scores, biasing attention toward nearby tokens.
  3. Sliding-window attention -- counting attention entries directly to
     show O(T*w) linear growth vs. full attention's O(T^2).

Run:
    python example.py
"""

import torch

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. RoPE: rotate Q/K by position, dot product depends only on (i - j)
# ---------------------------------------------------------------------------

def rope_rotate(x, position, base=10000.0):
    """x: (..., d), d even. Rotates each (x[2i], x[2i+1]) pair by an angle
    proportional to `position` and specific to that pair's frequency --
    exactly the per-pair rotation from Phase 02 Lesson 3's example.py,
    now applied to Q/K themselves instead of added to the input."""
    d = x.shape[-1]
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))   # (d/2,)
    angles = position * freqs                                      # (d/2,)
    cos, sin = torch.cos(angles), torch.sin(angles)

    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = x1 * cos - x2 * sin
    rotated[..., 1::2] = x1 * sin + x2 * cos
    return rotated


def rope_demo():
    print("=" * 70)
    print("1. RoPE: DOT PRODUCT DEPENDS ONLY ON RELATIVE POSITION")
    print("=" * 70)

    d = 8
    q = torch.randn(d)
    k = torch.randn(d)

    print("Fixed q, k. Rotating both by position, then dotting the results,")
    print("for several (pos_i, pos_j) pairs grouped by their DIFFERENCE:\n")

    test_pairs = [(0, 0), (10, 10), (50, 50),      # difference 0
                  (0, 5), (10, 15), (50, 55),        # difference -5
                  (5, 0), (20, 15), (100, 95)]        # difference +5

    for pos_i, pos_j in test_pairs:
        q_rot = rope_rotate(q, pos_i)
        k_rot = rope_rotate(k, pos_j)
        score = (q_rot @ k_rot).item()
        print(f"  pos_i={pos_i:4d}  pos_j={pos_j:4d}  (diff={pos_i - pos_j:+3d})  "
              f"rotated dot product = {score:.4f}")

    print("\n-> Every row with the SAME difference gives (up to floating point)")
    print("   the SAME score, regardless of the absolute positions involved.")
    print("   This is a mathematical guarantee of the rotation, not something")
    print("   the model has to learn -- unlike Phase 02's additive sinusoidal")
    print("   encoding, where a linear layer could in principle extract relative")
    print("   position, but nothing forces it to.")


# ---------------------------------------------------------------------------
# 2. ALiBi: bias attention scores directly by distance, no parameters
# ---------------------------------------------------------------------------

def alibi_slopes(num_heads):
    """The paper's geometric slope sequence for a power-of-2 number of heads."""
    start = 2 ** (-8 / num_heads)
    return torch.tensor([start ** (h + 1) for h in range(num_heads)])


def alibi_bias_matrix(T, slope):
    """bias[i, j] = -slope * (i - j) for j <= i (causal); -inf for j > i."""
    positions = torch.arange(T)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)   # (T, T), i - j
    bias = -slope * distance.float()
    causal_mask = torch.tril(torch.ones(T, T)).bool()
    bias = bias.masked_fill(~causal_mask, float("-inf"))
    return bias


def alibi_demo():
    print("\n" + "=" * 70)
    print("2. ALiBi: A FIXED, PARAMETER-FREE DISTANCE PENALTY")
    print("=" * 70)

    num_heads = 4
    slopes = alibi_slopes(num_heads)
    print(f"Geometric slopes for {num_heads} heads: {slopes.numpy().round(4)}")

    T = 6
    bias = alibi_bias_matrix(T, slope=slopes[0].item())
    print(f"\nALiBi bias matrix for head 0 (slope={slopes[0]:.4f}), T={T}:")
    print(bias.numpy().round(3))

    print("\nEffect on attention: start from IDENTICAL (zero) raw content scores")
    print("for every key -- i.e. content alone gives no preference at all --")
    print("and see what ALiBi's bias alone does to the softmax weights:")
    raw_scores = torch.zeros(T)   # content says "no preference"
    query_pos = T - 1             # last position attending backward
    biased_scores = raw_scores + bias[query_pos]
    weights = torch.softmax(biased_scores, dim=-1)
    print(f"  query position {query_pos}, attention weights over keys 0..{T - 1}:")
    print(f"  {weights.numpy().round(4)}")
    print("  -> Even with zero content signal, ALiBi alone concentrates attention")
    print("     on the NEAREST positions -- a built-in recency bias needing zero")
    print("     learned parameters, and well-defined for any sequence length.")


# ---------------------------------------------------------------------------
# 3. Sliding-window attention: linear vs. quadratic growth
# ---------------------------------------------------------------------------

def count_full_attention_entries(T):
    return T * (T + 1) // 2   # causal: query i attends to i+1 keys


def count_sliding_window_entries(T, w):
    total = 0
    for i in range(T):
        total += min(i + 1, w)   # query i attends to at most w preceding keys (incl. itself)
    return total


def sliding_window_demo():
    print("\n" + "=" * 70)
    print("3. SLIDING-WINDOW ATTENTION: O(T*w) vs. FULL ATTENTION'S O(T^2)")
    print("=" * 70)

    window = 8
    print(f"Window size w={window}\n")
    print(f"{'seq_len T':>10}{'full attention entries':>25}{'sliding-window entries':>25}"
          f"{'ratio (full/window)':>22}")
    for T in [16, 64, 256, 1024, 4096, 16384]:
        full = count_full_attention_entries(T)
        windowed = count_sliding_window_entries(T, window)
        print(f"{T:>10}{full:>25,}{windowed:>25,}{full / windowed:>22.1f}")

    print("\n-> Full attention's entry count grows QUADRATICALLY with T -- the ratio")
    print("   keeps climbing. Sliding-window attention's grows LINEARLY (it's")
    print("   bounded by T*w), so the gap between them widens dramatically at")
    print("   long sequence lengths -- exactly the compute savings that make")
    print("   long-context models with very large T practical at all.")


def main():
    rope_demo()
    alibi_demo()
    sliding_window_demo()


if __name__ == "__main__":
    main()

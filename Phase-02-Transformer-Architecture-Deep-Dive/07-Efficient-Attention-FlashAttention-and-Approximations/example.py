"""
Efficient Attention: FlashAttention, Sparse and Linear Attention

Four parts, all working from the same scaled dot-product attention formula
as Lesson 2 (Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V), applied to
plain 2D (T, d) tensors -- single sequence, single head -- since the point
of every technique here is the T-dependence of the COMPUTATION, not the
batch/multi-head bookkeeping Lesson 2 already covered.

  0. THE COST, MADE CONCRETE -- how big the T x T score matrix actually gets
     at real context lengths.

  PART A. FlashAttention -- an EXACT reformulation. `flash_attention_tiled`
     implements the online-softmax recurrence, processing K/V one block at
     a time so the full T x T score matrix is never materialized, and is
     verified numerically IDENTICAL to naive attention (up to floating
     point). NOTE: this is a pure-Python loop over ordinary PyTorch ops, not
     a fused kernel -- it demonstrates the ALGORITHM's exactness and its
     reduced peak-materialized-matrix size, not a real GPU wall-clock
     speedup (real FlashAttention's speed comes from a fused CUDA kernel
     keeping blocks resident in on-chip SRAM; a Python-level loop over the
     same tensor ops adds its own overhead and, as measured directly below,
     ends up SLOWER in wall-clock terms here despite being the
     memory-efficient algorithm on real hardware).

  PART B. Sparse attention -- a real APPROXIMATION. Restricts each query to
     a local window plus a handful of global tokens, cutting the number of
     (query, key) pairs computed. Unlike Part A, this changes the output.

  PART C. Linear attention -- a kernel feature map replacing softmax,
     giving true O(T) compute via phi(Q) @ (phi(K)^T @ V) instead of
     softmax(Q K^T) V. Verifies the causal recurrent (RNN-style,
     step-by-step state update) form matches the causal parallel
     (cumulative-sum) form exactly, then measures REAL wall-clock scaling
     of the non-causal parallel linear form against naive attention as T
     grows -- this one IS a genuine, measurable speed win in plain PyTorch,
     because it's fewer total floating point operations and no T x T
     matrix, not a Python-loop-vs-fused-kernel comparison like Part A.

Runtime: ~10-30 seconds on a CPU (Part C's scaling demo is the slowest
piece, up to T=2048).

Run:
    python example.py
"""

import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ===========================================================================
# 0. THE COST, MADE CONCRETE
# ===========================================================================

def cost_demo():
    print("=" * 78)
    print("0. THE COST OF FULL ATTENTION, MADE CONCRETE")
    print("=" * 78)
    print("Naive attention materializes one (T, T) score matrix PER HEAD, per layer.")
    print("Sizes below are for that score matrix alone, float32 (4 bytes/element),")
    print("summed across a realistic 32 attention heads, for a SINGLE layer:\n")

    print(f"{'context length T':>18}{'one head (T*T)':>20}{'32 heads, 1 layer':>22}")
    for T in [4_096, 32_768, 131_072]:
        one_head_bytes = T * T * 4
        all_heads_bytes = one_head_bytes * 32
        print(f"{T:>18,}{one_head_bytes / 1e9:>17.2f} GB{all_heads_bytes / 1e9:>19.2f} GB")

    print("\n-> Quadrupling T (4,096 -> 131,072, a realistic jump for long-context")
    print("   models) grows the score matrix by 32x, not 4x -- T^2 growth. This is")
    print("   the exact O(T^2) cost flagged and deferred back in Phase 01 Lesson 5,")
    print("   section 5 (the trade-off: quadratic complexity). Phase 03 Lesson 6 and")
    print("   Phase 09 Lesson 2 each attack adjacent costs (how far a model can")
    print("   usefully attend at all; redundant recomputation during generation) --")
    print("   this lesson attacks the raw per-pass O(T^2) computation directly.")


# ===========================================================================
# PART A: FLASHATTENTION -- EXACT, TILED, ONLINE-SOFTMAX ATTENTION
# ===========================================================================

def naive_attention(Q, K, V, causal=False):
    """The Lesson 2 / Lesson 6 formula, unchanged. Q, K: (T, d_k). V: (T, d_v)."""
    d_k = Q.shape[-1]
    T = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (T, T) -- the matrix Part A avoids materializing
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def flash_attention_tiled(Q, K, V, block_size, causal=False):
    """The online-softmax algorithm (README §3): sweep across K/V in blocks
    of `block_size`, maintaining a running max `m`, running sum `l`, and
    running (unnormalized) output accumulator `O`, rescaling the running
    accumulators by exp(old_max - new_max) every time a new block raises
    the max. The full (T, T) score matrix is never formed -- only one
    (T, block_size) slice exists at a time.

    Simplification vs. real FlashAttention: this tiles only over K/V blocks
    (the query side stays whole). Real FlashAttention also tiles the query
    dimension so every block pair fits in on-chip SRAM together -- but the
    online-softmax recurrence, which is the actual mathematical trick, is
    identical either way.
    """
    T, d_k = Q.shape
    d_v = V.shape[-1]

    O = torch.zeros(T, d_v)
    l = torch.zeros(T, 1)                    # running softmax denominator
    m = torch.full((T, 1), float("-inf"))    # running row-wise max score

    num_blocks = math.ceil(T / block_size)
    q_idx = torch.arange(T).unsqueeze(1)     # (T, 1), for the causal mask below

    for j in range(num_blocks):
        k_start, k_end = j * block_size, min((j + 1) * block_size, T)
        K_j = K[k_start:k_end]               # (block, d_k) -- only this slice is ever materialized
        V_j = V[k_start:k_end]

        scores_j = Q @ K_j.transpose(-2, -1) / math.sqrt(d_k)   # (T, block)
        if causal:
            k_idx = torch.arange(k_start, k_end).unsqueeze(0)   # (1, block)
            scores_j = scores_j.masked_fill(k_idx > q_idx, float("-inf"))

        m_j = scores_j.max(dim=-1, keepdim=True).values         # (T, 1)
        m_new = torch.maximum(m, m_j)

        # Rescale factor for everything accumulated so far. Guarded against
        # exp(-inf - -inf) = NaN, which would only arise for a row that has
        # seen ZERO valid keys in every block so far (not reachable by this
        # loop's construction -- see README -- but guarded defensively).
        alpha = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), torch.exp(m - m_new))
        p_j = torch.exp(scores_j - m_new)                       # (T, block)

        l = alpha * l + p_j.sum(dim=-1, keepdim=True)
        O = alpha * O + p_j @ V_j
        m = m_new

    return O / l


def part_a_demo():
    print("\n" + "=" * 78)
    print("PART A: FLASHATTENTION -- TILED, ONLINE-SOFTMAX ATTENTION (EXACT)")
    print("=" * 78)

    d_k, d_v, block_size = 16, 16, 16
    print(f"Verifying flash_attention_tiled matches naive_attention EXACTLY")
    print(f"(block_size={block_size}), across several sequence lengths (including")
    print(f"lengths that are NOT multiples of block_size, to stress-test edges):\n")

    print(f"{'T':>8}{'causal':>10}{'max abs diff':>16}")
    for T in [8, 17, 64, 100]:
        for causal in [False, True]:
            Q = torch.randn(T, d_k)
            K = torch.randn(T, d_k)
            V = torch.randn(T, d_v)

            out_naive = naive_attention(Q, K, V, causal=causal)
            out_flash = flash_attention_tiled(Q, K, V, block_size, causal=causal)

            max_diff = (out_naive - out_flash).abs().max().item()
            print(f"{T:>8}{str(causal):>10}{max_diff:>16.2e}")
            assert torch.allclose(out_naive, out_flash, atol=1e-5), \
                f"Mismatch at T={T}, causal={causal}"

    print("\n-> Every max abs diff above is at floating-point precision (~1e-6 or")
    print("   smaller), not just 'close' -- flash_attention_tiled computes the")
    print("   mathematically EXACT same attention output as naive_attention, for")
    print("   every T tested including sizes that don't divide evenly by")
    print("   block_size. Zero quality tradeoff, by construction.")

    print("\n" + "-" * 78)
    print("Peak intermediate matrix size actually held in memory at once:")
    print("-" * 78)
    print(f"{'T':>10}{'naive (T x T)':>20}{'tiled (T x block)':>22}")
    for T in [1_024, 4_096, 16_384]:
        naive_elems = T * T
        tiled_elems = T * block_size
        print(f"{T:>10,}{naive_elems:>20,}{tiled_elems:>22,}")
    print(f"\n-> With block_size={block_size} fixed, the tiled approach's peak matrix")
    print(f"   size grows LINEARLY with T (T * {block_size}); naive's grows QUADRATICALLY")
    print(f"   (T * T). This is the real memory-shape saving FlashAttention's tiling")
    print(f"   is built on -- in a real fused GPU kernel, that smaller per-block")
    print(f"   matrix is what stays resident in fast on-chip SRAM instead of ever")
    print(f"   touching slow HBM, which is where FlashAttention's real wall-clock")
    print(f"   speedup on actual hardware comes from.")

    print("\n" + "-" * 78)
    print("Honest wall-clock check: naive vs. this PURE-PYTHON tiled loop")
    print("-" * 78)
    T, d = 2048, 32
    Q, K, V = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(5):
            naive_attention(Q, K, V)
        naive_time = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        for _ in range(5):
            flash_attention_tiled(Q, K, V, block_size=128)
        tiled_time = (time.perf_counter() - t0) / 5

    print(f"naive_attention:        {naive_time * 1000:8.2f} ms  (one big matmul)")
    print(f"flash_attention_tiled:   {tiled_time * 1000:8.2f} ms  ({math.ceil(T / 128)} Python-level block iterations)")
    print(f"\n-> As documented at the top of this file: the tiled version is SLOWER")
    print(f"   here ({tiled_time / naive_time:.1f}x), not faster. That is expected and honest --")
    print(f"   this is a Python for-loop calling ordinary PyTorch ops once per block,")
    print(f"   which adds real per-iteration Python/dispatch overhead that a single")
    print(f"   large matmul doesn't pay. Real FlashAttention's speed advantage lives")
    print(f"   entirely inside a fused CUDA kernel that performs this exact same")
    print(f"   recurrence without ever leaving the GPU chip or paying Python overhead")
    print(f"   -- something no pure-Python loop over PyTorch tensor ops can reproduce.")
    print(f"   What this demo DOES prove for real: the algorithm is exact (above),")
    print(f"   and its peak materialized matrix shrinks from T*T to T*block_size.")


# ===========================================================================
# PART B: SPARSE ATTENTION -- LOCAL WINDOW + GLOBAL TOKENS
# ===========================================================================

def build_sparse_mask(T, window, num_global):
    """True = allowed to attend. Two kinds of allowed connections:
      - local: |i - j| <= window // 2 (a fixed-size neighborhood)
      - global: the first `num_global` positions can see, and be seen by,
        EVERY position -- a small number of "hub" tokens that let
        information still propagate across the whole sequence, the same
        role global tokens play in Longformer/BigBird.
    """
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    local = (i - j).abs() <= window // 2
    is_global_row = i < num_global
    is_global_col = j < num_global
    return local | is_global_row | is_global_col


def part_b_demo():
    print("\n" + "=" * 78)
    print("PART B: SPARSE ATTENTION -- LOCAL WINDOW + GLOBAL TOKENS (APPROXIMATE)")
    print("=" * 78)
    print("The local-window piece here is exactly Phase 03 Lesson 6 section 3's")
    print("sliding-window attention; global tokens (a la Longformer/BigBird) are")
    print("added on top so information can still reach across the whole sequence")
    print("through a small number of always-visible hub positions.\n")

    print(f"{'T':>8}{'window':>10}{'num_global':>12}{'allowed pairs':>16}{'full T*T':>14}{'fraction computed':>20}")
    for T, window, num_global in [(256, 16, 2), (1024, 32, 4), (4096, 64, 8)]:
        mask = build_sparse_mask(T, window, num_global)
        allowed = mask.sum().item()
        full = T * T
        print(f"{T:>8}{window:>10}{num_global:>12}{allowed:>16,}{full:>14,}{allowed / full:>20.2%}")

    print("\n-> As T grows with window/num_global held fixed, the fraction of pairs")
    print("   actually computed keeps shrinking -- this mask's cost grows as")
    print("   O(T * window), linear in T, not O(T^2).")

    print("\n" + "-" * 78)
    print("Unlike Part A, this genuinely changes the output (a real approximation):")
    print("-" * 78)
    T, d_k, d_v, window, num_global = 256, 16, 16, 32, 4
    Q, K, V = torch.randn(T, d_k), torch.randn(T, d_k), torch.randn(T, d_v)

    out_full = naive_attention(Q, K, V)

    mask = build_sparse_mask(T, window, num_global)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float("-inf"))
    out_sparse = F.softmax(scores, dim=-1) @ V

    mean_abs_diff = (out_full - out_sparse).abs().mean().item()
    are_close = torch.allclose(out_full, out_sparse, atol=1e-3)
    print(f"mean |full_attention_output - sparse_attention_output| = {mean_abs_diff:.4f}")
    print(f"outputs numerically equal (atol=1e-3)? {are_close}")
    print("\n-> A real, nonzero difference (and NOT numerically equal) -- sparse")
    print("   attention trades some of full attention's information (whatever a")
    print("   masked-out position could have contributed) for the O(T*window) cost")
    print("   above. Whether that trade is worth it depends entirely on how much a")
    print("   given task actually needs long-range, non-local, non-hub attention.")


# ===========================================================================
# PART C: LINEAR ATTENTION -- KERNEL FEATURE MAP, O(T) COMPUTE
# ===========================================================================

def phi(x):
    """The positive elementwise feature map (Katharopoulos et al., 2020)."""
    return F.elu(x) + 1.0


def linear_attention_parallel(Q, K, V):
    """Non-causal linear attention: phi(Q) @ (phi(K)^T @ V). Matrix
    multiplication is associative, so computing phi(K)^T @ V FIRST gives a
    (d_k, d_v) matrix -- independent of T -- and the whole thing costs
    O(T * d^2) instead of naive attention's O(T^2 * d). No (T, T) matrix is
    EVER formed here, unlike Part A (which avoids materializing it all at
    once but still computes and touches every one of its T^2 entries) --
    this genuinely never computes them."""
    phi_Q, phi_K = phi(Q), phi(K)
    KV = phi_K.transpose(-2, -1) @ V                       # (d_k, d_v) -- NOT (T, T)
    Z = phi_K.sum(dim=-2)                                   # (d_k,) normalizer
    numerator = phi_Q @ KV                                  # (T, d_v)
    denominator = phi_Q @ Z.unsqueeze(-1)                    # (T, 1)
    return numerator / denominator


def linear_attention_causal_parallel_cumsum(Q, K, V):
    """Causal linear attention, PARALLEL (cumulative-sum) form -- used at
    training time, when the whole sequence is available at once. Materializes
    a (T, d_k, d_v) tensor of running outer products via cumsum; fine for a
    small demo, but note this is NOT the constant-memory form (that's the
    recurrent version below) -- it's the training-time-parallel counterpart."""
    phi_Q, phi_K = phi(Q), phi(K)
    outer = phi_K.unsqueeze(-1) * V.unsqueeze(-2)            # (T, d_k, d_v), outer[t] = phi_K[t] (x) V[t]
    KV_cumulative = torch.cumsum(outer, dim=0)                # (T, d_k, d_v): KV_cumulative[t] = sum_{s<=t} outer[s]
    Z_cumulative = torch.cumsum(phi_K, dim=0)                  # (T, d_k)

    numerator = torch.einsum("td,tdv->tv", phi_Q, KV_cumulative)   # (T, d_v)
    denominator = (phi_Q * Z_cumulative).sum(dim=-1, keepdim=True)  # (T, 1)
    return numerator / denominator


def linear_attention_causal_recurrent(Q, K, V):
    """Causal linear attention, RECURRENT (step-by-step) form -- the
    constant-memory, RNN-equivalent form used at inference time. A running
    (d_k, d_v) state is updated one token at a time, exactly like an RNN's
    hidden state -- no growing KV cache of individual tokens needed at all,
    unlike ordinary softmax attention (Phase 09 Lesson 2)."""
    T, d_k = Q.shape
    d_v = V.shape[-1]
    phi_Q, phi_K = phi(Q), phi(K)

    state = torch.zeros(d_k, d_v)     # the running (d_k, d_v) state -- an RNN-style hidden state
    z = torch.zeros(d_k)              # running normalizer

    outputs = []
    for t in range(T):
        state = state + torch.outer(phi_K[t], V[t])    # state_t = state_{t-1} + phi(k_t) (x) v_t
        z = z + phi_K[t]
        out_t = (phi_Q[t] @ state) / (phi_Q[t] @ z)
        outputs.append(out_t)
    return torch.stack(outputs, dim=0)


def part_c_equivalence_demo():
    print("\n" + "=" * 78)
    print("PART C.1: LINEAR ATTENTION -- RECURRENT (RNN-style) == PARALLEL (cumsum)")
    print("=" * 78)

    T, d_k, d_v = 32, 8, 8
    Q, K, V = torch.randn(T, d_k), torch.randn(T, d_k), torch.randn(T, d_v)

    out_recurrent = linear_attention_causal_recurrent(Q, K, V)
    out_parallel = linear_attention_causal_parallel_cumsum(Q, K, V)
    max_diff = (out_recurrent - out_parallel).abs().max().item()

    print(f"T={T}: max abs diff between recurrent and parallel-cumsum forms: {max_diff:.2e}")
    assert torch.allclose(out_recurrent, out_parallel, atol=1e-5), "Recurrent/parallel forms disagree!"
    print("-> Numerically identical. The step-by-step state update really is just")
    print("   another way to compute the exact same cumulative sum -- causal linear")
    print("   attention IS an RNN, mathematically, not merely 'RNN-like.'")

    print("\nAlso confirming the NON-causal parallel form matches the causal one")
    print("with an all-True (no masking) equivalent -- i.e. it's the same")
    print("mechanism, just without the running/cumulative restriction to the past:")
    out_noncausal = linear_attention_parallel(Q, K, V)
    print(f"(non-causal) output shape: {tuple(out_noncausal.shape)}  "
          f"-- a DIFFERENT tensor from the causal outputs above by construction")
    print("(every position can see the whole sequence, not just positions <= t),")
    print("shown here only to confirm the function runs and shapes line up.")


def part_c_scaling_demo():
    print("\n" + "=" * 78)
    print("PART C.2: REAL WALL-CLOCK SCALING -- NON-CAUSAL LINEAR vs. NAIVE ATTENTION")
    print("=" * 78)
    print("Unlike Part A's Python-loop-vs-single-matmul comparison, this one IS a")
    print("fair apples-to-apples comparison: both sides here are ordinary PyTorch")
    print("matmuls, no Python-level looping over blocks on either side. The only")
    print("difference is which matmuls get computed -- O(T^2 d) vs. O(T d^2).\n")

    d = 32
    Ts = [128, 256, 512, 1024, 2048]
    print(f"{'T':>8}{'naive (ms)':>14}{'linear (ms)':>14}{'speedup':>10}")
    naive_times, linear_times = [], []
    for T in Ts:
        Q, K, V = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(10):
                naive_attention(Q, K, V)
            naive_t = (time.perf_counter() - t0) / 10

            t0 = time.perf_counter()
            for _ in range(10):
                linear_attention_parallel(Q, K, V)
            linear_t = (time.perf_counter() - t0) / 10

        naive_times.append(naive_t)
        linear_times.append(linear_t)
        print(f"{T:>8}{naive_t * 1000:>14.3f}{linear_t * 1000:>14.3f}{naive_t / linear_t:>9.1f}x")

    print("\nGrowth factor each time T doubles (naive should trend toward ~4x per")
    print("doubling -- O(T^2) -- linear should trend toward ~2x -- O(T)):")
    print(f"{'T -> 2T':>14}{'naive growth':>16}{'linear growth':>16}")
    for i in range(1, len(Ts)):
        naive_growth = naive_times[i] / naive_times[i - 1]
        linear_growth = linear_times[i] / linear_times[i - 1]
        print(f"{Ts[i-1]:>6} -> {Ts[i]:<5}{naive_growth:>16.2f}{linear_growth:>16.2f}")

    print("\n-> Timing noise means these growth factors won't land exactly on 4x/2x,")
    print("   but the TREND should be clearly visible: naive's per-doubling growth")
    print("   factor runs noticeably higher than linear attention's as T increases --")
    print("   the real, measurable consequence of O(T^2) vs. O(T) scaling, with no")
    print("   Python-loop overhead confound this time on either side.")


def main():
    cost_demo()
    part_a_demo()
    part_b_demo()
    part_c_equivalence_demo()
    part_c_scaling_demo()


if __name__ == "__main__":
    main()

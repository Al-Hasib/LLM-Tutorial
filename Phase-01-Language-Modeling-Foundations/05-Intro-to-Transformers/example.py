"""
Introduction to Transformers

Two small, concrete demos of the two claims made in the README:
  1. A minimal single-head self-attention layer from scratch (full
     deep dive with multi-head + masking + scaling justification comes
     in Phase 02) -- every position attends to every other position.
  2. A structural comparison of a sequential (RNN-like) computation vs.
     a parallel (self-attention-like) computation, timed directly, to
     make "parallelization is why Transformers won" concrete rather
     than just asserted.

Run:
    python example.py
"""

import time
import numpy as np

rng = np.random.default_rng(0)


def softmax_rows(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Minimal single-head self-attention
# ---------------------------------------------------------------------------

def self_attention(X, Wq, Wk, Wv):
    """X: (T, d_model). Every position attends to every position, including
    itself -- this is the full generalization of the Seq2Seq attention from
    the previous lesson, with queries, keys, AND values all coming from the
    same sequence X."""
    Q = X @ Wq   # (T, d_k)
    K = X @ Wk   # (T, d_k)
    V = X @ Wv   # (T, d_v)

    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)   # (T, T) -- scaling previewed here, derived in Phase 02
    weights = softmax_rows(scores)       # (T, T), each row sums to 1
    output = weights @ V                 # (T, d_v)
    return output, weights


def self_attention_demo():
    print("=" * 70)
    print("1. MINIMAL SELF-ATTENTION: EVERY TOKEN ATTENDS TO EVERY TOKEN")
    print("=" * 70)

    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    T, d_model, d_k = len(tokens), 12, 8

    X = rng.normal(size=(T, d_model))          # toy "embeddings" for each token
    Wq = rng.normal(scale=0.3, size=(d_model, d_k))
    Wk = rng.normal(scale=0.3, size=(d_model, d_k))
    Wv = rng.normal(scale=0.3, size=(d_model, d_k))

    output, weights = self_attention(X, Wq, Wk, Wv)

    print(f"tokens = {tokens}")
    print(f"input shape {X.shape} -> output shape {output.shape}")
    print("\nAttention weight matrix (rows = query token, columns = key token,")
    print("each row sums to 1 -- read row i as 'how much token i looks at every")
    print("other token'):\n")
    header = "        " + "".join(f"{t:>7s}" for t in tokens)
    print(header)
    for i, row in enumerate(weights):
        row_str = "".join(f"{w:7.3f}" for w in row)
        print(f"{tokens[i]:>7s} {row_str}")
    print("\nNote both repeated occurrences of 'the' (positions 0 and 4) get their")
    print("own independently computed attention row -- position alone doesn't")
    print("determine the pattern, content does (this is randomly initialized and")
    print("untrained, so the specific pattern isn't meaningful yet -- Phase 02")
    print("trains this end to end).")


# ---------------------------------------------------------------------------
# 2. Sequential (RNN-like) vs. parallel (attention-like) computation, timed
# ---------------------------------------------------------------------------

def sequential_processing(X, W):
    """Simulates an RNN: h_t depends on h_{t-1}, forcing a strict Python loop
    with T sequential steps that cannot be vectorized away."""
    T, d = X.shape
    h = np.zeros(d)
    for t in range(T):
        h = np.tanh(X[t] + h @ W)   # each step MUST wait for the previous one
    return h


def parallel_processing(X, Wq, Wk, Wv):
    """Simulates self-attention: one shot of matrix multiplications covering
    the WHOLE sequence at once -- no step has to wait for another."""
    output, _ = self_attention(X, Wq, Wk, Wv)
    return output


def timing_demo():
    print("\n" + "=" * 70)
    print("2. SEQUENTIAL (RNN-LIKE) vs. PARALLEL (ATTENTION-LIKE) COMPUTATION")
    print("=" * 70)
    print("Structural path length between the first and last token:")
    print(f"  {'seq_len':>8}  {'RNN hops needed':>16}  {'self-attn hops needed':>22}")
    for T in [10, 50, 200, 1000]:
        print(f"  {T:>8}  {T - 1:>16}  {1:>22}")
    print("  -> In an RNN, information from token 0 passes through T-1 sequential")
    print("     steps to reach the last token. In self-attention it's always a")
    print("     single dot product, no matter how far apart the tokens are.\n")

    d = 64
    print(f"{'seq_len':>8}  {'sequential time (s)':>20}  {'parallel time (s)':>18}")
    for T in [50, 200, 800, 3000]:
        X = rng.normal(size=(T, d))
        W = rng.normal(scale=0.1, size=(d, d))
        Wq = rng.normal(scale=0.1, size=(d, d))
        Wk = rng.normal(scale=0.1, size=(d, d))
        Wv = rng.normal(scale=0.1, size=(d, d))

        start = time.perf_counter()
        sequential_processing(X, W)
        seq_time = time.perf_counter() - start

        start = time.perf_counter()
        parallel_processing(X, Wq, Wk, Wv)
        par_time = time.perf_counter() - start

        print(f"{T:>8}  {seq_time:>20.6f}  {par_time:>18.6f}")

    print("\n-> Notice the 'parallel' version actually gets SLOWER than the")
    print("   sequential one as T grows large -- and that's an honest, important")
    print("   result, not a bug. Self-attention does O(T^2) total work (a full")
    print("   TxT score matrix), while the RNN loop does only O(T) work overall.")
    print("   On a single CPU core, more total work simply takes more time.")
    print("   The point was never that attention does LESS work -- it's that")
    print("   attention's T^2 dot products have NO dependencies between them, so")
    print("   a GPU with thousands of cores can compute all of them at once,")
    print("   while the RNN's T steps have a hard dependency chain and can only")
    print("   ever run one after another, however many cores you throw at it.")
    print("   This single-threaded benchmark can't reward that property -- it")
    print("   takes real parallel hardware to see the wall-clock win -- but it")
    print("   does make the O(T) vs. O(T^2) compute trade-off from the README")
    print("   completely concrete.")


def main():
    self_attention_demo()
    timing_demo()


if __name__ == "__main__":
    main()

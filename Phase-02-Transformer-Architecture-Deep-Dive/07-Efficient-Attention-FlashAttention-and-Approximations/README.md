# Efficient Attention: FlashAttention, Sparse and Linear Attention

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `07-Efficient-Attention-FlashAttention-and-Approximations`

## Why this matters

[Lesson 2](../02-Self-Attention-and-Multi-Head-Attention/README.md) built self-attention correctly but left its cost unaddressed: computing `Q @ K.T` materializes a full `(T, T)` matrix, quadratic in sequence length `T` in both compute AND memory — a cost flagged all the way back in [Phase 01 Lesson 5 §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity) and deferred ever since. Two other lessons in this course each attack an *adjacent* cost — [Phase 03 Lesson 6](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md) fixes how far a model can usefully generalize its *position* sense, and [Phase 09 Lesson 3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md) fixes redundant *recomputation* during autoregressive generation — but neither touches the raw `O(T²)` cost of a single full attention pass itself. This lesson finally does, with three genuinely different fixes: **FlashAttention**, which computes the exact same result far more efficiently on real hardware, and **sparse** and **linear attention**, which both change what gets computed at all in exchange for true sub-quadratic scaling.

## What this lesson covers

- Why self-attention is `O(T²)` in both time and memory — made concrete with real numbers
- FlashAttention: an exact, IO-aware reformulation — no quality tradeoff, just faster/leaner on real hardware
- The online-softmax trick that makes tiled, exact attention possible without ever materializing the full `(T, T)` matrix
- Sparse attention: restrict which positions can attend to which, cutting compute at the cost of some lost context
- Linear attention: replace softmax with a kernel feature map to get true `O(T)` compute, and its RNN-equivalent recurrent form
- An honest comparison of what's exact vs. approximate, and what production systems actually reach for

## 1. The cost, made concrete

Self-attention computes `scores = Q @ K.T`, shape `(T, T)`, then softmax, then `@ V`. Both the compute (`~T² · d`) and — just as importantly — the memory needed to hold that score matrix (`T²` floats) grow quadratically in `T`. `example.py` prints this out at real context lengths: even accounting for only ONE layer's worth of score matrices across 32 heads in float32, `T=4,096` needs about 2 GB, but `T=32,768` — a 8x longer context, entirely realistic for a modern long-context model — needs over 137 GB, and `T=131,072` needs over **2 terabytes**. Quadrupling the context length multiplies the cost by roughly 16-32x, not 4x. This is the concrete shape of the problem every technique below attacks.

## 2. FlashAttention (Dao et al., 2022): exact, but IO-aware

The key insight is **not** a different math result — FlashAttention computes bit-for-bit (up to floating-point rounding) the same attention output as the naive formula. The insight is that on a real GPU, attention's bottleneck usually isn't raw FLOPs but **memory bandwidth**: repeatedly writing and reading the full `(T, T)` score matrix to slow HBM (high-bandwidth memory — still far slower than the GPU's on-chip SRAM) dominates wall-clock time. FlashAttention tiles `Q`, `K`, `V` into blocks small enough to fit in fast on-chip SRAM and processes attention block by block, so the full `(T, T)` matrix is **never written to HBM at all**.

## 3. The online-softmax trick

Softmax needs a running max (for numerical stability) and a sum over **all** of `K`/`V` — but tiling means only one block of `K`/`V` is visible at a time. The fix: maintain a running max `m`, a running (rescaled) sum `l`, and a running (rescaled) output accumulator `O` while sweeping across blocks. Every time a new block's local max exceeds the running max, **rescale** everything accumulated so far by `exp(old_max - new_max)` before folding in the new block's contribution:

```
for each K/V block j:
    scores_j = Q @ K_j.T / sqrt(d_k)                 # (T, block_size) -- never (T, T)
    m_new    = max(m, rowmax(scores_j))
    alpha    = exp(m - m_new)                         # rescale factor for OLD accumulators
    p_j      = exp(scores_j - m_new)
    l        = alpha * l + rowsum(p_j)
    O        = alpha * O + p_j @ V_j
    m        = m_new
output = O / l
```

This computes the *exact* softmax incrementally, one block at a time, never holding more than one block's score matrix in memory at once. `example.py` implements this loop and verifies it against naive attention — including sequence lengths that don't divide evenly by the block size, and both causal and non-causal masking — and every run lands at floating-point precision (differences on the order of `1e-7`, not just "close"). It also prints the peak intermediate matrix each approach holds at once: naive's grows as `T × T`; the tiled version's stays `T × block_size`, linear in `T` for a fixed block size — the real memory-shape saving FlashAttention's tiling is built on.

**Honest caveat, stated directly in the script**: `example.py`'s tiled implementation is a plain Python `for` loop over ordinary PyTorch tensor ops, not a fused CUDA kernel — so measuring its wall-clock time against one big `naive_attention` matmul call actually shows the tiled version running *slower* here, purely from Python-level loop and dispatch overhead. That's expected and honest, not a bug: real FlashAttention's wall-clock advantage comes entirely from a fused kernel that keeps every block resident in on-chip SRAM without ever leaving the GPU or paying Python overhead — something no pure-Python loop over PyTorch ops can reproduce. What this lesson's code *does* prove for real: the algorithm is exact, and its peak materialized-matrix size genuinely shrinks from `T²` to `T · block_size`.

## 4. Sparse attention: change what gets computed

Unlike FlashAttention, this changes the *result*: restrict each query position to a fixed subset of key positions instead of all `T`. `example.py` combines two patterns:

- **Local window** — attend only within a fixed-size neighborhood. This is exactly [Phase 03 Lesson 6 §3's sliding-window attention](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md#3-sliding-window-local-attention-fixing-the-compute-not-just-position), already implemented there — nothing to re-derive, just reuse the idea.
- **Global tokens** (Longformer/BigBird-style) — a handful of designated positions that attend to, and are attended to by, *every* position, acting as hub points information can still route through across the whole sequence despite the local restriction elsewhere.

`example.py` builds this combined mask and counts, exactly, what fraction of the full `T²` pairs actually get computed: with a window/global-token count held fixed, that fraction keeps shrinking as `T` grows (single-digit percentages by `T` in the low thousands) — the concrete signature of `O(T · window)` cost, linear in `T`. Critically, and unlike Part 2-3's FlashAttention, running this masked attention on the same random `Q`/`K`/`V` as full attention produces a **genuinely different** output (a real, nonzero mean difference, not numerically equal at any reasonable tolerance) — sparse attention is a true approximation, trading away whatever a masked-out position could have contributed for its reduced cost.

## 5. Linear attention: true `O(T)` compute via a kernel feature map

Replace `softmax(Q K^T) V` with `phi(Q) @ (phi(K)^T @ V)` for an elementwise positive feature map `phi` (here, `phi(x) = elu(x) + 1`, from Katharopoulos et al., 2020). Because matrix multiplication is associative, computing `phi(K)^T @ V` **first** gives a `(d_k, d_v)` matrix — independent of `T` — so the whole computation costs `O(T · d²)` instead of `O(T² · d)`: linear in sequence length, and (unlike FlashAttention's tiling) a `(T, T)` matrix is never computed at all, not even one block at a time.

There's also a causal, **recurrent** equivalent form: a running `(d_k, d_v)` state gets updated one token at a time exactly like an RNN's hidden state — `state_t = state_{t-1} + phi(k_t) ⊗ v_t` — with no growing per-token KV cache needed at all, unlike ordinary softmax attention's cache ([Phase 09 Lesson 3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md)). `example.py` implements both the step-by-step recurrent form and a parallel cumulative-sum form and verifies they're numerically identical — causal linear attention **is** an RNN mathematically, not merely RNN-*like*.

Because this — unlike Part 2-3's tiling loop — is a genuine algorithmic reduction in the number of floating-point operations (a fair, apples-to-apples matmul-vs-matmul comparison, no Python-loop-vs-single-call confound), `example.py`'s timing comparison of non-causal linear attention against naive attention shows a **real, measurable** wall-clock advantage that grows with `T`: in one live run, linear attention starts roughly on par with naive attention at small `T` and pulls to a well-over-10x wall-clock advantage by `T=2,048`, with naive's per-doubling growth factor consistently landing well above linear attention's as `T` increases — the concrete, measured shape of `O(T²)` vs. `O(T)`. (Exact multipliers vary run to run with timing noise; the script prints its own live numbers rather than asserting fixed ones here.)

## 6. Comparison — and what production systems actually use

| | FlashAttention | Sparse attention | Linear attention |
|---|---|---|---|
| Changes the output? | No — exact | Yes — real approximation | Yes — different function entirely |
| Asymptotic cost | Still `O(T²)` compute, but IO-optimal | `O(T · window)` | `O(T · d²)`, linear in `T` |
| Typical use today | The default in virtually every real training/inference stack | Dedicated long-context / local-attention architectures (cross-link [Phase 03 Lesson 6](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md)) | Influential, but a real historical quality gap vs. softmax attention on language tasks kept it out of most top LLMs |

Linear attention's recurrent-state view — a fixed-size state updated one token at a time — is conceptually close to the state-space-model idea covered in [Phase 10 Lesson 3](../../Phase-10-Advanced-and-Frontier-Topics/03-State-Space-Models-Mamba/README.md): both replace attention's growing, all-pairs computation with a constant-size running state, and both trade away exactly the same thing (content-based, all-pairs selectivity) to get there. FlashAttention, meanwhile, is the odd one out on purpose: it's not a tradeoff at all, which is exactly why it — not sparse or linear attention — became the near-universal default.

## Video Script Outline

1. Motivation — Lesson 2 built attention correctly but never addressed its `O(T²)` cost; three different fixes, three different tradeoffs
2. The cost, in real gigabytes, at real context lengths
3. FlashAttention: exact output, IO-aware — the memory-bandwidth bottleneck, not FLOPs, is the real target
4. The online-softmax recurrence, worked through step by step
5. Walkthrough of `example.py` Part A — numerically exact verification against naive attention, the shrinking peak-matrix-size table, and the honest Python-loop-vs-fused-kernel wall-clock caveat
6. Sparse attention: local window (recap Phase 03 Lesson 6) + global tokens, a real approximation this time
7. Linear attention: the kernel-feature-map trick, and the recurrent form's RNN equivalence, proven numerically
8. Walkthrough of `example.py` Part C's real wall-clock scaling comparison, and the final three-way comparison table

## Further Reading

- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- Dao (2023), *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer*
- Zaheer, Guruganesh, Dubey, et al. (2020), *Big Bird: Transformers for Longer Sequences*
- Katharopoulos, Vyas, Pappas, Fleuret (2020), *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*
- Choromanski, Likhosherstov, Dohan, et al. (2021), *Rethinking Attention with Performers*

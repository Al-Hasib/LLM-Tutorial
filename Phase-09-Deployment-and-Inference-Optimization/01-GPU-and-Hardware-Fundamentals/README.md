# GPU and Hardware Fundamentals

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `01-GPU-and-Hardware-Fundamentals`

## Why this matters

Every lesson later in this phase quietly leans on a small set of hardware facts without ever stating them: [Lesson 2](../02-Quantization/README.md) says a smaller weight is often a *faster* weight; [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) says one decode step is "memory-bandwidth-bound, not compute-bound"; [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md) says FlashAttention's whole point is avoiding a round trip to "slow HBM." None of those claims is derived from anything — they're asserted, because the hardware reality underneath them is assumed knowledge. This lesson is that missing foundation: a small number of facts about how a GPU is built and how data moves on it, formalized once, so every later "this is memory-bound" or "this saves bandwidth" claim in this phase has real ground to stand on instead of being taken on faith.

## What this lesson covers

- GPU architecture at the level that actually matters for LLM inference: cores, Streaming Multiprocessors, warps — not a CUDA programming course
- The memory hierarchy: HBM vs. SRAM, and why the size/speed gap between them is the central fact of GPU performance
- Memory bandwidth as a hard, finite resource
- FLOPs: what they are, and how to count them for the operation that dominates a Transformer (matrix multiplication)
- The roofline model: one ratio — arithmetic intensity — that decides whether an operation is compute-bound or memory-bound
- The payoff: why LLM **prefill** is compute-bound and **decode** is memory-bound, and why that single fact explains half of this phase's later optimization tricks

## 1. GPU architecture, at the level that matters here

A GPU is built from many **Streaming Multiprocessors (SMs)** — tens to well over a hundred on a modern data-center chip — each of which runs large groups of threads in lockstep, called **warps** (32 threads executing the same instruction at once on different data). This is the opposite design point from a CPU: a CPU has a handful of complex, high-clock-speed cores individually optimized to race through a long, branchy, sequential chain of dependent instructions as fast as possible; a GPU has vastly more, much simpler compute units, individually slower and dumber, optimized for doing the *exact same instruction* to enormous numbers of independent data points *simultaneously*. Matrix multiplication — the operation that dominates every layer of a Transformer (the `Q`/`K`/`V` projections, the FFN, the output head; recall [Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) and [Phase 02 Lesson 5](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#5-the-position-wise-feed-forward-network-ffn)) — is exactly this kind of workload: the same multiply-accumulate operation, repeated over enormous numbers of independent output elements. That match is the entire reason GPUs (not CPUs) run LLMs.

## 2. The memory hierarchy: HBM vs. SRAM

Two very different kinds of memory sit on a GPU. **HBM** (High Bandwidth Memory) is the large pool — tens of gigabytes on a modern data-center card — that holds the model's weights, activations, and KV cache; it is fast in absolute terms, but slow *relative to the compute units sitting right next to the chip*. **SRAM** — small, fast on-chip memory local to each SM (sometimes called shared memory), often only a few megabytes in total across the whole chip — is dramatically faster to access, but far too small to hold a model's weights. Every operation's inputs have to make the trip from HBM into SRAM (and from there into registers) before an SM can compute anything with them, and results have to make the trip back out. This gap — small-and-fast vs. large-and-slower — is the single most important fact about GPU performance, and it's exactly what [Phase 02 Lesson 7&#39;s FlashAttention](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md#2-flashattention-dao-et-al-2022-exact-but-io-aware) exploits for attention specifically: by tiling `Q`/`K`/`V` into blocks that fit in SRAM, it avoids ever writing the full attention score matrix out to HBM at all. This lesson is the general version of that same HBM/SRAM gap; that lesson is its most famous worked example.

## 3. Memory bandwidth as a hard resource

Moving bytes between HBM and the compute units happens at a fixed, finite rate — **memory bandwidth**, measured in GB/s or TB/s. A modern data-center GPU offers very high bandwidth by everyday standards (on the order of a couple of terabytes per second for a high-end accelerator), but it is still a hard ceiling: every operation must pay this cost, in time, to read its inputs from HBM and write its outputs back, no matter how fast the compute units themselves could in principle work. Bandwidth and compute speed are two *separate* hardware specs, and — as the next two sections make precise — which one actually limits a given operation's speed depends entirely on what that operation is doing with the data once it has it.

## 4. FLOPs: counting the actual arithmetic

A **FLOP** is one floating-point operation (one add, or one multiply). For a matrix multiply of an `(m, k)` matrix by a `(k, n)` matrix, producing an `(m, n)` output, each of the `m*n` output elements is a dot product over `k` terms — `k` multiplies and `k-1` (~`k`) adds — so the total cost is:

```
FLOPs(matmul) = 2 * m * k * n
```

**Peak FLOPs** is a fixed hardware spec: the maximum number of these operations per second the compute units could physically perform, if they never had to wait for data. It is a completely separate number from memory bandwidth (§3) — a chip's peak FLOPs and peak bandwidth are two independent limits, and an operation can only ever run as fast as whichever limit it actually hits.

## 5. The roofline model: arithmetic intensity decides compute-bound vs. memory-bound

Define **arithmetic intensity** for an operation as how much arithmetic it does per byte of data it has to move:

```
arithmetic intensity = FLOPs performed / bytes moved from memory
```

Every accelerator has a **ridge point** — `peak FLOPs / peak memory bandwidth` — the arithmetic intensity at which the compute limit and the memory-bandwidth limit are exactly balanced. An operation with arithmetic intensity **above** the ridge point is **compute-bound**: enough arithmetic gets done per byte moved that the compute units stay the bottleneck, memory bandwidth is not fully saturating them, and a faster chip (more peak FLOPs) would genuinely speed it up. An operation **below** the ridge point is **memory-bound**: the compute units sit idle, waiting for bytes to arrive, and no amount of extra peak FLOPs would help at all — only more bandwidth (or moving fewer bytes) would. This single ratio is the entire content of the "roofline model" (Williams, Waterman & Patterson, 2009) — plot achievable performance against arithmetic intensity, and every real operation falls somewhere on a graph shaped like a sloped roof rising to a flat ceiling, with the ridge point exactly at the corner.

## 6. The payoff: why prefill is compute-bound and decode is memory-bound

This is the single most important application of everything above to LLM inference, and it's the concept [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) builds directly on. LLM inference has two named phases: **prefill** — the one parallel forward pass that processes the entire input prompt at once and produces the first output token — and **decode** — each subsequent forward pass that produces exactly one more token, fed by the token just generated. Look at what each one asks the same weight matrix to do:

- **Prefill** multiplies a whole prompt's worth of activations — `seq_len` tokens, potentially thousands — against each weight matrix in one matmul. The weight matrix is read from HBM once, and its bytes get reused across every one of those `seq_len` tokens' worth of arithmetic: high arithmetic intensity, well above the ridge point, **compute-bound**. This is exactly why prefill throughput scales with a chip's raw FLOPs, and why batching several prompts' prefill together helps further.
- **Decode** multiplies just *one* token's worth of activations against the exact same weight matrix. The *entire* weight matrix (plus, with a KV cache, the growing cache — [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md)) still has to be read from HBM, but now for only one token's worth of arithmetic: arithmetic intensity collapses to nearly the theoretical floor, far below the ridge point, **memory-bound**. `example.py` computes this difference for real, concrete Transformer-sized matrices below — decode's arithmetic intensity lands almost exactly at the "one FLOP-pair per byte" floor a single-token matmul cannot beat, while prefill's lands orders of magnitude higher.

This is *the* reason decode latency is dominated by how fast weights can stream out of HBM, not by the chip's peak FLOPs — and it is the entire justification for batching multiple *decode* requests together (many tokens' worth of decode work sharing one weight read), which is exactly what continuous batching in [Lesson 4](../04-Serving-Frameworks/README.md) exists to do. `example.py` also sweeps decode arithmetic intensity across batch size directly, showing concretely how many concurrent decode requests it takes to push decode back above the ridge point.

## Video Script Outline

1. Motivation — every later lesson in this phase asserts "memory-bound" or "saves bandwidth" without ever proving it; this lesson proves it
2. GPU architecture: SMs and warps, why massive simple parallelism beats a few complex cores for matmul-shaped work
3. HBM vs. SRAM: the size/speed gap, with FlashAttention (Phase 02 Lesson 7) as the concrete worked example
4. Memory bandwidth as a hard ceiling, independent of compute speed
5. FLOPs: counting the arithmetic in a matmul, and peak FLOPs as a separate hardware spec
6. The roofline model: arithmetic intensity, the ridge point, compute-bound vs. memory-bound in one picture
7. Walkthrough of `example.py` Part A — real arithmetic-intensity numbers for prefill-shaped vs. decode-shaped matmuls, classified against a real GPU's ridge point, and the batch-size sweep showing where decode crosses back over
8. Walkthrough of `example.py` Part B — the CPU-measurable analogy, and recap into Lessons 2-4's optimizations, all now standing on real ground

## Further Reading

- Williams, Waterman, Patterson (2009), *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (the original roofline model)
- Dao, Fu, Ermon, Rudra, Ré (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — revisited from [Phase 02 Lesson 7](../../Phase-02-Transformer-Architecture-Deep-Dive/07-Efficient-Attention-FlashAttention-and-Approximations/README.md), the concrete worked example of the HBM/SRAM gap this lesson generalizes
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* — general reference on memory hierarchies and throughput-oriented processor design
- NVIDIA's public GPU architecture documentation (e.g. the Ampere/Hopper architecture whitepapers) for real, up-to-date peak-FLOPs and HBM-bandwidth specifications of specific chips

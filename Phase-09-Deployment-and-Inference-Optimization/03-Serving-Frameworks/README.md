# Serving Frameworks

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `03-Serving-Frameworks`

## Why this matters

[Lesson 1](../01-Quantization/README.md) shrank the *model*, and [Lesson 2](../02-KV-Cache-and-Speculative-Decoding/README.md) shrank the *work* each generation step has to do (the KV cache, speculative decoding). This lesson is about the layer above both of those: the actual serving software that takes a trained (possibly quantized) model, manages many concurrent users' KV caches in memory, and decides how to batch requests together on real hardware. Even a perfectly optimized model can be served wastefully — this lesson covers the three ideas that turned "run a model" into "run a model *efficiently, for many users at once*": vLLM's memory management, Hugging Face TGI's request scheduling, and llama.cpp's GPU-free deployment path. [Lesson 5](../05-Cost-and-Latency-Optimization/README.md) builds directly on the batching ideas here to reason about cost and latency trade-offs at the fleet level.

## What this lesson covers

- Why naive KV-cache memory allocation wastes enormous amounts of accelerator memory
- vLLM's PagedAttention: OS-style virtual memory paging applied to the KV cache
- Static batching's throughput problem with variable-length generations
- Hugging Face TGI's continuous (in-flight) batching
- llama.cpp and GGUF: serving without a GPU at all
- `example.py`: a discrete-event simulation proving continuous batching's throughput advantage with real numbers

## 1. The problem: naive KV-cache allocation wastes memory

Recall from [Lesson 2](../02-KV-Cache-and-Speculative-Decoding/README.md) that autoregressive generation caches every token's K and V vectors so they don't need recomputation at each step. A naive serving implementation allocates one **large, contiguous** buffer per sequence, sized for the *maximum possible* sequence length the server supports (say, 4096 tokens) — even if a particular request only ever generates 30 tokens before hitting an end-of-sequence token.

This wastes memory two ways:

- **Internal fragmentation**: most of each sequence's reserved buffer sits empty for its entire lifetime.
- **Inability to grow safely**: if two sequences are placed back-to-back in memory and one needs more room than expected, there is nowhere for it to grow without copying.

Because GPU memory is the hard limit on how many sequences can be served *concurrently* (batch size), wasted KV-cache memory directly caps throughput — it is the single biggest inefficiency naive serving code has.

## 2. vLLM's PagedAttention

Kwon et al. (2023) borrow an idea straight from operating-systems virtual memory: instead of one contiguous buffer per sequence, the KV cache is divided into small, fixed-size **blocks** ("pages"), typically enough for 16 tokens each. A per-sequence **block table** maps logical token positions to physical blocks, which can live *anywhere* in a shared pool of physical memory — not necessarily contiguous, not necessarily reserved in advance:

```
Naive:     [ sequence A: reserved for 4096 tokens, uses 30 ]  <- 4066 tokens wasted
           [ sequence B: reserved for 4096 tokens, uses 800 ] <- 3296 tokens wasted

PagedAttention:  physical block pool (shared, fixed-size blocks)
                 sequence A block table -> [ block 7 ]                (30 tokens -> 1 block)
                 sequence B block table -> [ block 2, block 9, ... ]  (800 tokens -> 50 blocks)
```

Blocks are allocated **on demand**, one at a time, as a sequence actually grows — exactly like a paging OS allocates physical pages to a process's virtual address space lazily. This eliminates internal fragmentation almost entirely (the only waste is inside the *last, partially-filled* block of each sequence) and means the number of sequences that fit in memory is limited by *actual* total tokens generated, not by a worst-case reservation. In practice this lets vLLM pack dramatically more concurrent sequences into the same GPU memory, which — combined with continuous batching below — is the source of vLLM's large reported throughput gains over naive Hugging Face `transformers` serving. PagedAttention also enables cheap **memory sharing**: if many requests share an identical prompt prefix (e.g. a system prompt), their block tables can point at the *same* physical blocks (copy-on-write), avoiding redundant KV-cache storage entirely — a preview of the prefix-caching idea in [Lesson 5](../05-Cost-and-Latency-Optimization/README.md).

## 3. Static batching's throughput ceiling

Even with efficient memory, *how* requests are grouped into a batch matters. The simplest scheme, **static batching**, forms a fixed-size batch of `N` requests and runs them together step by step until every one of them is finished, only then admitting the next `N` waiting requests:

```
batch = [req_1 (needs 10 tokens), req_2 (needs 200 tokens), req_3 (needs 15 tokens), req_4 (needs 180 tokens)]
```

Because real requests need wildly different numbers of output tokens (a one-line answer vs. a long explanation), the whole batch is only as fast as its **slowest member**. `req_1` and `req_3` finish after 10-15 steps but their batch slots sit **idle**, doing no useful work, until `req_2` and `req_4` finally finish around step 200 — at which point the *next* batch of 4 waiting requests can finally start, even though 2 of the 4 slots had been free for 185 steps.

## 4. Hugging Face TGI: continuous (in-flight) batching

TGI (and the underlying idea from Orca, Yu et al. 2022) fixes exactly this: instead of a batch being a fixed, static group, the server maintains a fixed number of **concurrent slots**, and the moment any slot's sequence finishes, that slot is **immediately backfilled** with the next request waiting in the queue — no waiting for the rest of the batch:

```
Static:      [====req_1====][xxxxxxxxxxxxxxxxxxxx idle xxxxxxxxxxxxxxxxxxxx]
             [====req_2==========================================================]
             (next batch cannot start until req_2's slot is free too)

Continuous:  [====req_1====][====req_5====][==req_8==][...]     <- slot immediately reused
             [====req_2==========================================================]
```

This keeps every slot doing useful work almost all the time regardless of how skewed the distribution of generation lengths is, which is exactly the workload real chat/completion traffic produces (some replies are one sentence, some are pages long). `example.py` builds a real discrete-event simulation of both schemes on an identical workload and measures the resulting difference in total time and slot utilization directly.

## 5. llama.cpp: serving without a GPU

llama.cpp (Gerganov et al.) takes a different axis entirely: instead of maximizing GPU throughput, it targets running LLMs efficiently on **CPUs and consumer hardware** with no GPU required at all. It loads models in **GGUF**, a single-file format storing quantized weights (tying directly back to [Lesson 1](../01-Quantization/README.md)'s INT8/INT4 quantization) plus tokenizer and metadata, and implements hand-optimized quantized matrix-multiply kernels for CPU SIMD instruction sets. This is the path that makes it possible to run a 7-13B parameter model on a laptop or phone — a fundamentally different deployment target than vLLM/TGI's multi-GPU, many-concurrent-user data-center serving, but built on the exact same quantization ideas from Lesson 1.

## 6. Choosing between them

| Framework | Optimizes for | KV-cache strategy | Typical deployment target |
|---|---|---|---|
| vLLM | Max throughput, many concurrent users | PagedAttention (paged, on-demand) | Multi-GPU data center serving |
| Hugging Face TGI | Utilization under variable-length traffic | Continuous batching (+ can combine with paged memory) | GPU data center serving |
| llama.cpp | Running at all without a GPU | Simple contiguous cache, but a tiny quantized model to begin with | Laptop / CPU / edge / phone |

None of these are mutually exclusive concepts — modern serving stacks (including TGI itself) increasingly combine paged memory management *and* continuous batching *and* quantized weights simultaneously; the frameworks above are just where each idea was first popularized at scale.

## Video Script Outline

1. Motivation — a perfectly optimized model can still be served wastefully; this lesson is about the serving software layer itself
2. The naive contiguous KV-cache buffer problem and why it caps concurrent batch size
3. vLLM's PagedAttention — OS-style paging applied to the KV cache, block tables, on-demand allocation, prefix sharing
4. Static batching's throughput ceiling — the whole batch waits for its slowest member
5. Hugging Face TGI's continuous/in-flight batching — immediate slot backfill, tie back to Orca
6. llama.cpp and GGUF — the GPU-free deployment path, tying back to Lesson 1's quantization
7. Walkthrough of `example.py` — a real discrete-event simulation of static vs. continuous batching on identical variable-length workloads, with measured total time and utilization
8. Recap + pointer to Lesson 5's cost/latency trade-offs, which build on today's batching ideas

## Further Reading

- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper)
- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models* (origin of continuous/iteration-level batching, which Hugging Face TGI implements)
- Gerganov et al., the `llama.cpp` project and the GGUF file format specification
- Hugging Face, *Text Generation Inference* (TGI) documentation

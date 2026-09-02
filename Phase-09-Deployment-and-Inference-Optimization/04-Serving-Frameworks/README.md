# Serving Frameworks

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `04-Serving-Frameworks`

## Why this matters

[Lesson 2](../02-Quantization/README.md) shrank the *model*, and [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) shrank the *work* each generation step has to do (the KV cache, speculative decoding). This lesson is about the layer above both of those: the actual serving software that takes a trained (possibly quantized) model, manages many concurrent users' KV caches in memory, and decides how to batch requests together on real hardware. Even a perfectly optimized model can be served wastefully — this lesson covers the ideas that turned "run a model" into "run a model *efficiently, for many users at once*": vLLM's memory management, Hugging Face TGI's request scheduling, SGLang's more general prefix sharing, NVIDIA TensorRT-LLM's compiled kernels, and llama.cpp's GPU-free deployment path. [Lesson 6](../06-Cost-and-Latency-Optimization/README.md) builds directly on the batching ideas here to reason about cost and latency trade-offs at the fleet level.

## What this lesson covers

- Why naive KV-cache memory allocation wastes enormous amounts of accelerator memory
- vLLM's PagedAttention: OS-style virtual memory paging applied to the KV cache
- Static batching's throughput problem with variable-length generations
- Hugging Face TGI's continuous (in-flight) batching
- Chunked prefill: fixing the head-of-line blocking a long prompt causes inside continuous batching
- SGLang's RadixAttention: automatic, general-purpose prefix sharing
- TensorRT-LLM: trading portability for a compiled, kernel-fused execution model
- llama.cpp and GGUF: serving without a GPU at all
- `example.py`: discrete-event simulations proving continuous batching's throughput advantage, and chunked prefill's latency-spike fix, with real numbers

## 1. The problem: naive KV-cache allocation wastes memory

Recall from [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) that autoregressive generation caches every token's K and V vectors so they don't need recomputation at each step. A naive serving implementation allocates one **large, contiguous** buffer per sequence, sized for the *maximum possible* sequence length the server supports (say, 4096 tokens) — even if a particular request only ever generates 30 tokens before hitting an end-of-sequence token.

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

Blocks are allocated **on demand**, one at a time, as a sequence actually grows — exactly like a paging OS allocates physical pages to a process's virtual address space lazily. This eliminates internal fragmentation almost entirely (the only waste is inside the *last, partially-filled* block of each sequence) and means the number of sequences that fit in memory is limited by *actual* total tokens generated, not by a worst-case reservation. In practice this lets vLLM pack dramatically more concurrent sequences into the same GPU memory, which — combined with continuous batching below — is the source of vLLM's large reported throughput gains over naive Hugging Face `transformers` serving. PagedAttention also enables cheap **memory sharing**: if many requests share an identical prompt prefix (e.g. a system prompt), their block tables can point at the *same* physical blocks (copy-on-write), avoiding redundant KV-cache storage entirely — a preview of the prefix-caching idea in [Lesson 6](../06-Cost-and-Latency-Optimization/README.md).

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

## 5. Chunked prefill: fixing continuous batching's head-of-line blocking

Continuous batching (§4) solves *decode*-side idling, but introduces a new problem the instant a long prompt arrives: processing a request's **prefill** — the single forward pass over its entire prompt, potentially thousands of tokens — is normally dispatched as one atomic step. If the scheduler runs that atomic prefill in the same slot-tick as everyone else's ongoing **decode** steps (one new token each), every one of those other, already-in-flight requests is stalled for however long the whole prefill takes — a real latency spike for interactive users who were mid-conversation, moments after continuous batching was supposed to guarantee them a slot every tick. This is **head-of-line blocking**: one long unit of work at the front of the queue delays everything behind it, even work that would otherwise have been instant.

**Chunked prefill** (the Sarathi line of work, Agrawal et al.) fixes this directly: split a long prefill into smaller chunks (a few hundred tokens each), and have the scheduler interleave one chunk of prefill with a normal decode step for every other in-flight request, every round — instead of either one long uninterrupted prefill, or pure decode-only rounds:

```
Atomic prefill (blocks everyone else for the FULL prefill length):
  round 1:  [=========== prefill: all 2000 tokens in one shot ===========]   <- other slots FROZEN this whole time
  round 2:  [decode][decode][decode]...                                      <- other slots resume

Chunked prefill (interleaved, chunk size 256):
  round 1:  [prefill chunk 1/8][decode][decode][decode]...   <- other slots only wait ~256 tokens' worth
  round 2:  [prefill chunk 2/8][decode][decode][decode]...
  ...
  round 8:  [prefill chunk 8/8][decode][decode][decode]...   <- long request's prefill now fully done too
```

Every round now mixes a little of both workloads: a chunk of prefill (compute-bound, large matmuls over many tokens at once) and a handful of other requests' decode steps (memory-bandwidth-bound, one token each) — smoothing out the worst-case per-step latency every request in the batch experiences, at the cost of a small, real overhead for the request being chunked (splitting and resuming its own prefill isn't perfectly free). `example.py` §3 simulates exactly this trade-off with real measured numbers: the worst-case delay chunking spares other in-flight requests from, and the small extra total time the chunked request itself pays for it.

## 6. SGLang: automatic, general-purpose prefix sharing

SGLang's (Zheng et al., 2024) headline serving idea is **RadixAttention**: a strictly more general version of the prefix-sharing PagedAttention enables (§2). vLLM's copy-on-write prefix sharing works when requests share one *pre-designated* prefix, like a fixed system prompt. RadixAttention instead organizes every sequence's cached KV blocks into a shared **radix tree**, keyed by the actual token sequences seen so far — so *any* two requests that happen to share *any* common prefix, discovered automatically rather than configured in advance, share the corresponding cached blocks, with the least-recently-used branches of the tree evicted automatically under memory pressure, the same way an LRU cache would. This generalizes prefix caching from "works for the one prefix you told the server about" to "works for whatever prefixes actually show up in real traffic." SGLang also introduces a structured-generation-oriented programming model — interleaving generation with explicit constraints and control flow — as a second, distinguishing feature beyond serving performance; that's the same constrained/structured-generation territory covered in depth in [Phase 07 Lesson 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/05-Structured-Output-and-Function-Calling/README.md), not re-derived here.

## 7. TensorRT-LLM: compiled, kernel-fused inference

NVIDIA's TensorRT-LLM takes a fundamentally different execution model from vLLM, TGI, and SGLang, all of which run a general Python serving loop that dispatches ordinary framework (e.g. PyTorch) operations. TensorRT-LLM instead **compiles** a specific model, ahead of time, into a highly optimized graph of hand-fused CUDA kernels targeting the *exact* GPU it will run on — trading portability (a compiled artifact is tied to one model, one precision, and one target GPU generation, and must be recompiled to move to another) for the maximum single-GPU, single-vendor throughput that only comes from kernel fusion and compile-time specialization a general Python loop can't match. It has converged on the same high-level ideas as the others — in-flight (continuous) batching and paged KV-cache management are both supported — the real distinguishing choice is compiled kernel graph vs. general serving loop, not which scheduling ideas each framework uses; these frameworks share ideas far more than they compete on inventing new ones.

## 8. llama.cpp: serving without a GPU

llama.cpp (Gerganov et al.) takes a different axis entirely: instead of maximizing GPU throughput, it targets running LLMs efficiently on **CPUs and consumer hardware** with no GPU required at all. It loads models in **GGUF**, a single-file format storing quantized weights (tying directly back to [Lesson 2](../02-Quantization/README.md)'s INT8/INT4 quantization) plus tokenizer and metadata, and implements hand-optimized quantized matrix-multiply kernels for CPU SIMD instruction sets. This is the path that makes it possible to run a 7-13B parameter model on a laptop or phone — a fundamentally different deployment target than vLLM/TGI's multi-GPU, many-concurrent-user data-center serving, but built on the exact same quantization ideas from Lesson 2.

## 9. Choosing between them

| Framework        | Optimizes for                                    | KV-cache strategy                                                  | Typical deployment target       |
| ---------------- | ------------------------------------------------- | -------------------------------------------------------------------| -------------------------------- |
| vLLM              | Max throughput, many concurrent users             | PagedAttention (paged, on-demand)                                  | Multi-GPU data center serving    |
| Hugging Face TGI  | Utilization under variable-length traffic         | Continuous batching (+ can combine with paged memory)              | GPU data center serving          |
| SGLang            | Automatic prefix reuse across arbitrary traffic   | RadixAttention (radix-tree paged cache, LRU eviction)               | GPU data center serving, structured-generation workloads |
| TensorRT-LLM      | Maximum single-GPU throughput on NVIDIA hardware  | Paged cache + in-flight batching, inside a compiled kernel graph    | Latency/throughput-critical NVIDIA deployments |
| llama.cpp         | Running at all without a GPU                      | Simple contiguous cache, but a tiny quantized model to begin with  | Laptop / CPU / edge / phone      |

None of these are mutually exclusive concepts — modern serving stacks increasingly combine paged memory management *and* continuous (or chunked-prefill-aware) batching *and* quantized weights simultaneously; the frameworks above are just where each idea was first popularized at scale, or, for TensorRT-LLM, the one framework that trades the others' portability for compiled, hardware-specific speed.

## Video Script Outline

1. Motivation — a perfectly optimized model can still be served wastefully; this lesson is about the serving software layer itself
2. The naive contiguous KV-cache buffer problem and why it caps concurrent batch size
3. vLLM's PagedAttention — OS-style paging applied to the KV cache, block tables, on-demand allocation, prefix sharing
4. Static batching's throughput ceiling — the whole batch waits for its slowest member
5. Hugging Face TGI's continuous/in-flight batching — immediate slot backfill, tie back to Orca
6. Chunked prefill — head-of-line blocking made concrete, and the interleaved-chunk fix
7. SGLang's RadixAttention — automatic prefix sharing via a radix tree, generalizing PagedAttention's prefix caching
8. TensorRT-LLM — compiled, kernel-fused execution vs. a general serving loop, the portability/speed trade-off
9. llama.cpp and GGUF — the GPU-free deployment path, tying back to Lesson 2's quantization
10. Walkthrough of `example.py` — discrete-event simulations of static vs. continuous batching, and of chunked vs. atomic prefill's head-of-line-blocking fix, all with measured numbers
11. Recap + pointer to Lesson 6's cost/latency trade-offs, which build on today's batching ideas

## Further Reading

- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper)
- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models* (origin of continuous/iteration-level batching, which Hugging Face TGI implements)
- Agrawal et al., the Sarathi / Sarathi-Serve line of work on piggybacking decode steps with chunked prefill to fix head-of-line blocking in continuous batching
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention and structured generation)
- NVIDIA, *TensorRT-LLM* documentation and GitHub repository
- Gerganov et al., the `llama.cpp` project and the GGUF file format specification
- Hugging Face, *Text Generation Inference* (TGI) documentation

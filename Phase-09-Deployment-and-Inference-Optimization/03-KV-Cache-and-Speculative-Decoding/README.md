# KV Cache and Speculative Decoding

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `03-KV-Cache-and-Speculative-Decoding`

## Why this matters

[Lesson 2](../02-Quantization/README.md) shrank the model itself. This lesson attacks a different cost entirely: the sheer number of redundant computations a Transformer does during **autoregressive generation**. Every LLM in this course generates text one token at a time, feeding each new token back in to predict the next ([Phase 02 Lesson 6&#39;s mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#4-autoregressive-generation)) — and naively, that means re-running causal self-attention ([Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md)) over the *entire* sequence so far, from scratch, at every single step. The **KV cache** is the single most important systems-level optimization in LLM inference, and it is the exact piece of machinery that [Phase 03 Lesson 7&#39;s discussion of Grouped-Query Attention](../../Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md#4-grouped-query-attention-gqa-a-new-practically-important-variant) already forward-referenced when it explained why GQA's smaller K/V projections matter. This lesson also covers **speculative decoding**, a complementary trick that speeds up generation further without changing a single output token. Both techniques are the foundation the serving frameworks in [Lesson 4](../04-Serving-Frameworks/README.md) are built around.

## What this lesson covers

- Why naive autoregressive generation recomputes the same attention work over and over
- **Prefill** and **decode**: the two named phases every LLM inference request goes through
- The KV cache: caching each token's Key and Value vectors so each new step only costs O(T), not O(T^2)
- How Grouped-Query Attention (recap) shrinks the size of that cache
- Speculative decoding: using a cheap draft model to propose tokens a big model verifies in parallel
- Why speculative decoding speeds things up without changing the output distribution at all
- **TTFT and ITL**: the two per-request latency metrics prefill and decode each produce

## 1. Recap: why self-attention is O(T^2)

Self-attention ([Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md)) computes, for a sequence of length `T`, a full `T x T` matrix of attention scores:

```
scores = (Q @ K^T) / sqrt(d_k)        # (T, T) -- every query attends to every key
```

Producing this matrix costs O(T^2) dot products, and generating that matrix from scratch at every position is unavoidable when processing an entire sequence **in parallel** during training (the whole point of [Phase 02 Lesson 6 §3](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction): one forward pass, T training signals for free). But generation is different: at inference time there is no ground-truth sequence to feed in all at once — tokens must be produced one at a time, sequentially.

## 2. The naive (and wasteful) way to generate

Every LLM inference request goes through two distinct, standardly-named phases. **Prefill** is the first: the entire input prompt is fed through the model in one parallel forward pass, and the model's last-position output is sampled to produce the very first output token — this is one big matmul per layer, all prompt tokens at once, exactly the "process a whole sequence in parallel" mode [Phase 02 Lesson 6 §3](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction) already used for training. **Decode** is everything after: each subsequent forward pass produces exactly one more token, fed by the token generated at the previous step, repeated until the response is done. [Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) shows, with real arithmetic-intensity numbers, why these two phases sit on opposite sides of the compute-bound/memory-bound line — prefill's one big matmul is compute-bound, decode's one-token-at-a-time matmuls are memory-bound; that distinction is assumed context for the rest of this lesson.

The most obvious way to implement decode given a growing sequence `x[0..t]` is: run the *entire* sequence through the model again, take the last position's output, sample the next token, append it, and repeat ([Phase 02 Lesson 6 §4](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#4-autoregressive-generation) does exactly this for simplicity). The problem: at step `t`, tokens `0..t-1` have **already been through this exact computation** at step `t-1`. Their Key and Value vectors haven't changed at all — only the newly appended token needs new K/V vectors, and only *one* new row of the attention-score matrix needs computing (the new token's query against everything before it — causal masking means nothing new needs to attend *to* the new token yet, since it doesn't exist for earlier positions). Recomputing the whole `T x T` score matrix from scratch every decode step means the total work summed over generating `T` tokens is O(T^3) — the O(T^2) cost of one attention pass, paid again at every one of the `T` generation steps.

## 3. The KV cache: pay for each token's K/V exactly once

The fix is to **cache** the Key and Value vectors for every token as soon as they're computed, and reuse them at every later step instead of recomputing them:

```
step t:  compute Q_t, K_t, V_t for ONLY the new token
         append K_t, V_t to the cache  ->  cache now holds K_0..K_t, V_0..V_t
         attend:  scores_t = (Q_t @ cache_K^T) / sqrt(d_k)     # (1, t+1), not (t+1, t+1)
         output_t = softmax(scores_t) @ cache_V
```

Each generation step now does O(t) work (one query against `t` cached keys) instead of O(t^2) work (recomputing the full matrix up to position `t`). Summed over generating `T` tokens, total work drops from O(T^3) to O(T^2) — the same total cost as a single ordinary forward pass over the full sequence, which is the best any causal-attention model could hope to do. This is a pure engineering optimization: the *mathematical* output of attention at every position is identical with or without the cache (`example.py` verifies this directly by checking that a naive and a cached generation loop produce byte-for-byte identical token sequences). The cost is memory, not compute: every layer must keep its own K/V cache in memory, growing linearly with sequence length and batch size.

## 4. Recap: Grouped-Query Attention shrinks the cache

[Phase 03 Lesson 7 §4](../../Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md#4-grouped-query-attention-gqa-a-new-practically-important-variant) already showed the punchline: the size of this cache scales directly with the number of distinct K/V projections a model has, i.e. `num_kv_heads`, not `num_heads`. Ordinary multi-head attention caches one K/V pair per head; Grouped-Query Attention shares one K/V pair across a *group* of heads, and Multi-Query Attention shares a single K/V pair across *all* heads. The cache-size formula from that lesson:

```
KV cache bytes = 2 * batch * seq_len * num_kv_heads * d_k * num_layers * bytes_per_value
```

Everything in this lesson's KV-cache mechanics applies unchanged to GQA/MQA models — only `num_kv_heads` in that formula changes; the caching *algorithm* (append new K/V, reuse the rest) is identical.

## 5. Speculative decoding: verify several tokens for the price of one pass

Even with a KV cache, generation is still fundamentally sequential: one new token per forward pass of the (large, expensive) model, and each forward pass has a fixed overhead regardless of how few new tokens it produces (memory-bandwidth-bound, not compute-bound, at small batch sizes — most of the model's weights must simply be read from memory once per step). **Speculative decoding** (Leviathan, Kalman & Matias, 2023; independently, Chen et al., 2023) breaks the "one token per big-model pass" assumption:

```
1. A small, fast "draft" model proposes K candidate next tokens autoregressively
   (K cheap forward passes through the SMALL model).
2. The large "target" model verifies ALL K candidates in a SINGLE parallel forward
   pass (feeding all K draft tokens at once, same trick as parallel training).
3. Walk through the K candidates left to right. Accept a candidate if a rejection-
   sampling test passes (comparing the target model's and draft model's
   probabilities for that token); reject at the first mismatch.
4. On rejection, sample one token from a corrected distribution derived from the
   target model at that position, discard the rest of the draft, and start the
   next round from there.
```

## 6. Why this is exact, and why it's faster

The rejection-sampling rule in step 3 is the crucial detail: it is constructed so that the *marginal* distribution over accepted tokens is provably identical to what the target model would have produced sampling token-by-token on its own — speculative decoding is a pure speed optimization, not an approximation, exactly like the KV cache. It costs nothing in output quality.

The speedup comes from **amortization**: verifying `K` draft tokens costs the target model *one* forward pass (parallel verification is roughly as cheap as generating one token, since the memory-bandwidth bottleneck dominates regardless of how many tokens are scored at once), but if several of those `K` tokens are accepted, the target model has effectively produced multiple tokens for the price of one expensive pass. The draft model's forward passes are cheap because it's small. The net win depends entirely on the **acceptance rate**: a draft model that agrees with the target often lets many tokens through per round; a draft model that disagrees constantly falls back to one-token-at-a-time target generation, which is no worse (up to a small constant overhead) than not using speculative decoding at all. `example.py` measures this directly: the number of expensive target-model calls needed to generate a fixed-length sequence, with vs. without speculative decoding, across a few draft/target agreement rates.

## 7. TTFT and ITL: the two per-request latency metrics

Prefill and decode each produce their own standard latency metric. **TTFT (Time to First Token)** is the latency from when a request arrives to when its first output token is produced — dominated by the prefill phase's compute time (plus whatever time the request spent queued behind other work), since nothing can be returned to the user until that first parallel forward pass over the whole prompt finishes. **ITL (Inter-Token Latency)** is the latency between each subsequent pair of output tokens — the cost of one decode step, dominated by the memory-bandwidth cost of reading the model's weights (and the growing KV cache) once per step, exactly [Lesson 1's](../01-GPU-and-Hardware-Fundamentals/README.md) memory-bound regime. Together they compose into total request latency:

```
total latency  ≈  TTFT  +  (num_output_tokens - 1) * average ITL
```

A long prompt inflates TTFT (more prefill compute before the first token); a long response inflates total latency mostly through the `ITL` term, one decode step at a time. Speculative decoding (§5-6 above) is specifically an **ITL** optimization — it doesn't touch prefill or TTFT at all, only the per-step cost of decode. These two per-request numbers are the fine-grained building blocks underneath the aggregate throughput/latency trade-off [Lesson 6](../06-Cost-and-Latency-Optimization/README.md) covers at the fleet level: a serving system's *aggregate* throughput and *per-request* latency are just TTFT and ITL, measured across many concurrent requests instead of one (Pope et al., 2022, already cited below, is a good reference for exactly this decomposition at serving scale).

## Video Script Outline

1. Motivation — "generation is sequential; can we stop redoing work we've already done?"
2. Prefill vs. decode: the two named phases, and why Lesson 1 puts them on opposite sides of the compute-bound/memory-bound line
3. Naive regeneration: why re-running the whole sequence every decode step is O(T^3) in total
4. The KV cache: cache K/V, append at each step, O(T) per step / O(T^2) total
5. Recap: GQA/MQA as a direct lever on the cache-size formula
6. Speculative decoding: draft proposes, target verifies in parallel, longest-correct-prefix accepted
7. Why the rejection-sampling scheme keeps the output distribution exactly unchanged
8. Walkthrough of `example.py` — matching outputs with/without cache, real attention-computation counts, and target-model-call savings from speculative decoding
9. TTFT and ITL: naming the two per-request latency numbers prefill and decode each produce, and how speculative decoding specifically targets ITL
10. Recap + pointer to [Lesson 4: Serving Frameworks](../04-Serving-Frameworks/README.md), where the KV cache becomes the resource that PagedAttention manages efficiently

## Further Reading

- Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*
- Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling*
- Pope et al. (2022), *Efficiently Scaling Transformer Inference* (KV cache memory/bandwidth analysis at serving scale)
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*

# Frontier Inference Systems

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `11-Frontier-Inference-Systems`

## Why this matters

This lesson is the capstone of the whole phase — not a new independent trick, but the place where every earlier lesson's idea gets pushed past the point where a single server or a single, uniform batching policy is enough. [Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) established that prefill is compute-bound and decode is memory-bound; [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) gave the KV cache and the GQA/MQA formula for shrinking it; [Lesson 4](../04-Serving-Frameworks/README.md) gave PagedAttention and automatic prefix sharing inside one server; [Lesson 6](../06-Cost-and-Latency-Optimization/README.md) gave the cost/latency trade-offs of batching and prefix caching; and [Lesson 10](../10-Production-Serving-and-Benchmarking/README.md) gave fleet-scale routing and observability across many replicas. The largest real-world serving deployments — the ones actually running frontier models at massive concurrent scale — take every one of those ideas and stretch it further: instead of interleaving two opposite-shaped workloads on one GPU, split them onto separate, specialized pools of hardware; instead of one server's local prefix cache, route across an entire fleet with cache-topology awareness; instead of a uniform fleet of identical accelerators, deliberately mix hardware generations to match each workload's resource profile; and instead of shrinking the KV cache only by sharing K/V across heads (GQA/MQA), shrink it far further by not caching per-head K/V at all. None of this is a fixed, finished recipe — it is where the field's engineering effort is concentrated right now.

## What this lesson covers

- Why continuous batching's single-GPU interleaving of prefill and decode is fundamentally a compromise, not a solution
- Prefill/decode disaggregation: separate GPU pools per phase, and the real cost of transferring the KV cache between them
- KV-cache-aware routing at fleet scale: extending RadixAttention-style prefix sharing from one server to an entire load-balanced fleet
- Heterogeneous serving: matching GPU generation to workload shape (and to cost) rather than assuming a uniform fleet
- Multi-head Latent Attention (MLA): compressing the KV cache to a single low-dimensional latent per token instead of per-head K/V
- How a real frontier serving stack composes quantization, MLA/GQA, PagedAttention, disaggregation, heterogeneous hardware, and fleet routing all at once

## 1. Recap: continuous batching forces prefill and decode to share one GPU

[Lesson 1 §6](../01-GPU-and-Hardware-Fundamentals/README.md#6-the-payoff-why-prefill-is-compute-bound-and-decode-is-memory-bound) established the core tension this whole lesson resolves: prefill's one big matmul over an entire prompt is compute-bound, while decode's one-token-at-a-time matmuls are memory-bound — two workloads with opposite resource profiles, running on the same weight matrices, on the same GPU. [Lesson 4](../04-Serving-Frameworks/README.md)'s continuous batching (§4 there) keeps decode slots full, and chunked prefill (§5 there) keeps a long prompt's prefill from stalling everyone else's decode steps — both are clever ways to interleave the two workloads on a *single* GPU without one starving the other. But interleaving is still a compromise: a GPU provisioned to be good at compute-bound work (more raw FLOPs) is never simultaneously the ideal choice for memory-bandwidth-bound work, and vice versa. Every chunk of prefill squeezed into a round of decode steps is, however small, still time that GPU spent not doing the thing it does best. The interleaving trick hides the tension; it doesn't remove it.

## 2. Prefill/decode disaggregation: separate pools, a real network cost

The more radical fix is to stop sharing a GPU between the two phases at all. **Prefill/decode disaggregation** splits prefill and decode onto physically *separate* pools of GPUs, each provisioned and tuned for its own workload — for example, fewer but more compute-heavy chips for the prefill pool (where raw FLOPs are the bottleneck), and more numerous, cheaper, memory-bandwidth-favorable chips for the decode pool (where FLOPs are mostly wasted anyway, per §1). A request's prefill runs entirely on the prefill pool, producing the first output token and a completed KV cache for the prompt; that KV cache must then be **transferred over the network** to a machine in the decode pool, which continues generation from there, token by token, entirely independent of whatever the prefill pool is doing for other requests.

```
Interleaved (single pool):     [prefill chunk][decode][decode][prefill chunk][decode]...
                                one GPU pool, one scheduling policy, workloads compete for the same hardware

Disaggregated (two pools):     prefill pool:  [=== prefill ===] --KV cache transfer--> decode pool
                                decode pool:                                            [decode][decode][decode]...
                                each pool scheduled and scaled independently, on hardware suited to its own workload
```

This removes the interleaving compromise of §1 entirely: the prefill pool can be scaled and scheduled purely around compute-bound throughput, the decode pool purely around memory-bandwidth and batching many concurrent decode streams, and neither one's scheduling policy has to compromise for the other's sake. Be honest about the real cost, though — this is a genuine trade-off, not a free win. The KV cache being transferred can be large (recall [Lesson 3 §3-4](../03-KV-Cache-and-Speculative-Decoding/README.md#3-the-kv-cache-pay-for-each-tokens-kv-exactly-once)'s memory formula: it scales with `batch * seq_len * num_kv_heads * d_k * num_layers`), so a long prompt means a real, non-trivial amount of data has to cross the network between two physical machines before decode can even begin — an added latency cost the single-pool design never had. There is also real added system complexity: two pools to schedule and keep balanced (instead of one), a network transfer path that has to be fast and reliable, and a harder capacity-planning problem (how many prefill machines vs. how many decode machines, given a traffic mix that can shift over time) than a single homogeneous pool ever posed.

## 3. KV-cache-aware routing at fleet scale

[Lesson 4 §6](../04-Serving-Frameworks/README.md#6-sglang-automatic-general-purpose-prefix-sharing)'s RadixAttention shares cached prefixes *within* one server's memory, automatically, via a radix tree; [Lesson 6 §2](../06-Cost-and-Latency-Optimization/README.md#2-prefix-caching-paying-for-a-shared-prompt-once) covers the economics of not paying for a shared prompt's prefill twice. Both of those ideas assume the request in question has already landed on the one server (or, after §2, the one decode-pool machine) that happens to have the relevant prefix cached. At fleet scale, with many independent replicas — or, once prefill and decode are disaggregated, separate pools each with many machines — that assumption stops being free: the **load balancer itself** has to decide which replica a new request goes to, and a naive policy has no way to know which replica already holds a given prefix in its cache.

[Lesson 10 §3](../10-Production-Serving-and-Benchmarking/README.md#3-load-balancing-across-replicas)'s round-robin and least-outstanding-requests policies are both blind to cache state by construction — they balance load, not cache locality. Route a request to a replica that has never seen its prefix before, and that replica pays the full, uncached prefill cost all over again, exactly the redundant cost §2 (via Lesson 6) exists to avoid — except now the redundancy is a routing accident, not a caching-layer limitation. A **cache-aware router** fixes this by tracking, at the fleet level, which replica most recently served (and therefore likely still has cached) each known prefix, and preferring that replica for any new request sharing the same prefix — falling back to an ordinary load-balancing policy only when no replica has a relevant prefix cached, or when the replica that does is already overloaded. This is the same prefix-sharing idea as §4/§6 of Lesson 4, just lifted one layer up: instead of one server's local radix tree, the router maintains (an approximation of) the cache *topology* of the entire fleet, and uses it as a first-class input to the routing decision itself, not an afterthought.

## 4. Heterogeneous serving: matching hardware to workload shape

Nothing about a serving fleet requires every GPU in it to be the same generation, or even the same role. §1-2 already introduced the idea that prefill and decode want different resource profiles; **heterogeneous serving** takes that idea and applies it directly to real, mixed hardware rather than assuming a uniform fleet of identical accelerators. A concrete pattern: put newer, pricier, higher-peak-FLOPs chips in the prefill pool, where §1's compute-bound workload directly benefits from more raw FLOPs per dollar of newer silicon — and put older, cheaper chips (retained rather than retired, or simply bought because they're cheaper per GPU) in the decode pool, where the workload is memory-bandwidth-bound and largely insensitive to how many peak FLOPs a chip has, so long as its memory bandwidth and capacity are adequate.

This ties directly to [Lesson 6](../06-Cost-and-Latency-Optimization/README.md)'s cost-engineering framing: matching hardware to workload shape is a **cost lever**, not just a performance one. Buying the newest, most expensive accelerator for every role in the fleet wastes money on decode machines that will never be compute-bound enough to use that chip's extra FLOPs, exactly as buying only cheap, bandwidth-constrained chips would starve a compute-bound prefill pool. A heterogeneous fleet is the disaggregation idea of §1-2 taken to its logical conclusion: not just *separate* pools, but pools built from *different* hardware, each chosen for the specific bottleneck (compute vs. bandwidth) its workload actually hits.

## 5. MLA: Multi-head Latent Attention shrinks the cache further than GQA/MQA can

[Lesson 3 §4](../03-KV-Cache-and-Speculative-Decoding/README.md#4-recap-grouped-query-attention-shrinks-the-cache) gave the KV-cache-size formula:

```
KV cache bytes = 2 * batch * seq_len * num_kv_heads * d_k * num_layers * bytes_per_value
```

GQA and MQA shrink this formula by reducing `num_kv_heads` — fewer *distinct* K/V projections, shared across groups of query heads (or, for MQA, across all of them). This helps, but it's a blunt instrument: every head in a shared group is forced to attend using literally identical K/V vectors, trading away some representational diversity for a smaller cache.

**Multi-head Latent Attention** (MLA, introduced with DeepSeek-V2, DeepSeek-AI 2024) takes a different, more aggressive approach. Instead of caching separate — even if shared — K/V vectors per head at all, MLA caches a single, much lower-dimensional **latent vector** per token per layer, and reconstructs the full, per-head K/V on the fly at attention time via learned per-head up-projection matrices:

```
GQA/MQA cache:  store num_kv_heads separate (d_k-dimensional) K/V vectors per token per layer
                -> attend directly against the cached K/V, no reconstruction needed

MLA cache:      store ONE d_latent-dimensional latent vector per token per layer  (d_latent << num_kv_heads * d_k)
                -> at attention time: up-project the latent, per head, into full-sized K/V on the fly
                   K_head_i = W_up_K_i @ latent   (and similarly for V), THEN attend as usual
```

The resulting cache-size formula replaces the `num_kv_heads * d_k` term with a single small `d_latent`:

```
MLA cache bytes = 2 * batch * seq_len * d_latent * num_layers * bytes_per_value
```

Because `d_latent` is typically far smaller than even *one* GQA group's worth of K/V (`d_k`), let alone `num_kv_heads * d_k`, MLA's cache can end up meaningfully smaller than what even aggressive GQA or MQA settings reach — while the per-head up-projection preserves (approximately) full multi-head expressiveness, rather than forcing a whole group of heads to share identical, un-differentiated K/V the way GQA does. Be honest about the trade-off, though: reconstructing per-head K/V from the latent at *every* attention computation adds a small amount of extra compute — the up-projection matmul — that GQA/MQA's directly-cached-and-reused K/V simply doesn't need to pay. MLA trades a little extra decode-time compute for a substantially smaller memory footprint; whether that trade is worth it depends on whether a deployment is memory-capacity-constrained (very long contexts, very large batches) or already comfortably within its memory budget. `example.py` §1 computes real cache-memory numbers for MHA, GQA, MQA, and MLA side by side, across a sweep of context lengths, on one concrete illustrative model configuration.

## 6. Recap: how a frontier serving stack composes all of this

A real frontier serving stack rarely uses just one of this phase's ideas — it composes nearly all of them simultaneously. Quantized weights ([Lesson 2](../02-Quantization/README.md)) and an MLA or GQA architecture ([Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) / §5 above) shrink the KV cache from two independent directions at once. PagedAttention and continuous batching ([Lesson 4](../04-Serving-Frameworks/README.md)) manage that (now smaller) cache efficiently and keep GPU slots full *within* each pool. Prefill and decode run on disaggregated pools (§2), possibly built from heterogeneous hardware chosen for each pool's specific bottleneck (§4). A cache-aware fleet router (§3) ties the pools together, sending each request to wherever its prefix is already cached and its target pool has capacity, all under the observability and autoscaling layer [Lesson 10](../10-Production-Serving-and-Benchmarking/README.md) provides across the whole fleet. No single idea here does all the work — the win comes from stacking cheaper weights, a smaller cache, smarter memory management, specialized hardware, and cache-aware routing on top of each other. And none of it is finished: disaggregation ratios, cache-aware routing algorithms, and attention variants like MLA are all still active, fast-moving research and engineering areas, not a settled, final architecture — the frontier keeps moving because every one of these levers still has real room left to push.

## Video Script Outline

1. Motivation — this is the capstone lesson: every earlier idea in the phase, pushed past what one server or one uniform policy can do
2. Recap: continuous batching interleaves prefill and decode on one GPU, but interleaving is a compromise, not a fix, for their opposite resource profiles
3. Prefill/decode disaggregation: separate pools, independent scaling, and the honest cost of a real KV-cache network transfer
4. KV-cache-aware routing: lifting RadixAttention-style prefix sharing from one server's memory to an entire fleet's load balancer
5. Heterogeneous serving: newer compute-heavy chips for prefill, cheaper bandwidth-favorable chips for decode, framed as a cost lever
6. MLA: from GQA/MQA's shared-K/V-per-group idea to a single latent vector per token, reconstructed via per-head up-projection at attention time
7. Walkthrough of `example.py` §1 — real computed KV-cache-memory numbers for MHA vs. GQA vs. MQA vs. MLA across a context-length sweep
8. Walkthrough of `example.py` §2 — a Monte Carlo simulation of cache-aware vs. round-robin fleet routing, and the real hit-rate gap between them
9. Recap: how quantization, MLA/GQA, PagedAttention, disaggregation, heterogeneous hardware, and fleet routing all compose in a real frontier stack
10. Closing note: these are actively evolving systems and research areas, not a finished recipe

## Further Reading

- DeepSeek-AI (2024), *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (introduces Multi-head Latent Attention)
- Zhong et al. (2024), *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving* (prefill/decode disaggregation as a serving architecture)
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* — already cited in [Lesson 4 §6](../04-Serving-Frameworks/README.md#6-sglang-automatic-general-purpose-prefix-sharing); RadixAttention is the single-server prefix-sharing idea §3 above extends to fleet scale
- Pope et al. (2022), *Efficiently Scaling Transformer Inference* — already cited in [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) and [Lesson 6](../06-Cost-and-Latency-Optimization/README.md); the throughput/latency analysis underneath disaggregated and heterogeneous serving at scale
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — already cited in [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md); the cache-size baseline MLA is contrasted against in §5 above

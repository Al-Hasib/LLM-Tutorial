# Cost and Latency Optimization

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `05-Cost-and-Latency-Optimization`

## Why this matters

Every earlier lesson in this phase attacked cost and latency from the *model* or *system* side: [quantization](../01-Quantization/README.md) shrinks the weights, [KV caching and speculative decoding](../02-KV-Cache-and-Speculative-Decoding/README.md) cut redundant compute during generation, [serving frameworks](../03-Serving-Frameworks/README.md) squeeze more throughput out of a fixed batch of hardware, and [distillation and pruning](../04-Model-Distillation-and-Pruning/README.md) produce smaller models that are cheaper to run in the first place. This lesson is the capstone: it steps back to the level of the whole *serving system and traffic pattern*, and asks how to combine everything already built into an actual cost/latency budget. Two ideas dominate real production systems here — the **batching trade-off** (a direct consequence of [Lesson 3's](../03-Serving-Frameworks/README.md#2-continuous-in-flight-batching-hugging-face-tgi) batching discussion) and **routing cheap models at cheap models' problems** (which is exactly why small distilled or pruned models from [Lesson 4](../04-Model-Distillation-and-Pruning/README.md) are worth having around at all — not to replace the large model everywhere, but to *absorb the easy fraction of traffic*). This closes out Phase 09; [Phase 10](../../Phase-10-Advanced-and-Frontier-Topics/README.md) picks back up with frontier research topics that build on this entire deployment toolkit.

## What this lesson covers

- The throughput vs. latency trade-off inherent to batching
- Prefix caching: reusing a shared prompt's KV cache across many requests
- Model routing and cascades: sending easy queries to cheap models, hard queries to expensive ones
- How these three techniques compose into one cost/latency budget

## 1. The throughput vs. latency trade-off

[Lesson 3](../03-Serving-Frameworks/README.md#2-continuous-in-flight-batching-hugging-face-tgi) established *why* batching exists: running several sequences through the model together shares one forward pass's fixed overhead (kernel launches, weight loads from memory) across many requests, so the *hardware-level* cost per request drops as batch size grows. But that same fixed-size batch has a cost nobody pays for free: a **static** batch cannot start until enough requests have queued up to fill it, and every request in that batch is stuck waiting for the *last* request to be ready before the batch can even begin —

```
per-request latency ~= (time waiting for the batch to fill) + (time for the whole batch to be processed)
```

The first term grows with batch size `B` (more requests needed before the batch is full means a longer average wait), while the fixed processing cost per batch stays roughly constant thanks to parallel hardware — so the batch's **throughput ceiling** (`B / batch_processing_time`) rises with `B` even as each individual request's **wait** rises too. There is no batch size that is simultaneously best for both metrics: a service that must guarantee low per-request latency (interactive chat) wants small batches or continuous batching ([Lesson 3 §2](../03-Serving-Frameworks/README.md#2-continuous-in-flight-batching-hugging-face-tgi)); a service processing a large backlog of requests where nobody is watching a spinner (nightly batch summarization jobs) wants the largest batch the hardware can hold. `example.py` §1 simulates this trade-off directly with real, randomly-arriving requests and reports both numbers across a range of batch sizes.

## 2. Prefix caching: paying for a shared prompt once

Many real workloads repeatedly send the *same* long prefix with a different short suffix per request — a fixed system prompt, a fixed set of few-shot examples, or a long document that many different questions get asked against. Recall from [Lesson 2](../02-KV-Cache-and-Speculative-Decoding/README.md#1-the-kv-cache) that the KV cache is nothing more than the saved key/value vectors from a forward pass over some prefix of tokens. If many requests share an identical prefix, **that prefix's KV cache can be computed once and reused** across every request that shares it, instead of being recomputed from scratch for each one:

```
Without prefix caching: cost per request  = prefill(shared_prefix) + prefill(unique_suffix) + decode
With prefix caching:     cost per request  = prefill(unique_suffix) + decode      (shared_prefix cost paid ONCE, amortized)
```

For a workload where the shared prefix is much longer than the unique suffix (a long system prompt or a large retrieved document, with a short user question appended), this can eliminate the majority of the prefill compute for every request after the first. This is precisely the systems idea behind vLLM's prefix-caching feature and behind the "prompt caching" features shipped by major hosted LLM providers (Anthropic, OpenAI, and others) — see the provider documentation in Further Reading. It composes directly with [Lesson 3's PagedAttention](../03-Serving-Frameworks/README.md#1-vllms-pagedattention), which is precisely the mechanism that makes it cheap to let many sequences *share* pages of a cached prefix's KV blocks without copying them.

## 3. Model routing and cascades

Not every query needs the largest, most expensive model available. A **cascade** routes each incoming query to one of several models of increasing cost and capability, escalating only when necessary:

```
query -> cheap_model_or_classifier decides: "easy" or "hard"?
    "easy"  -> serve with the small/cheap model  (low cost)
    "hard"  -> escalate to the large/expensive model (higher cost, but only paid when needed)
```

The routing decision itself needs to be cheap relative to the savings it produces — typically either (a) a small, separately-trained classifier that predicts query difficulty from cheap features (length, topic, a quick embedding), or (b) simply the small model's *own* confidence in its answer (e.g., the entropy or max-probability of its output distribution) used as a proxy for whether escalation is warranted. Because most real-world query distributions are skewed toward easy/common cases, routing even a modest fraction of "hard" queries up to the expensive model can capture most of the expensive model's accuracy while paying its cost only for the minority of traffic that actually needs it. The risk is asymmetric and must be tuned deliberately: a cascade with a systematically overconfident cheap model will silently serve wrong answers for hard queries it thinks are easy — the accuracy loss from a leaky cascade doesn't show up in cost, only in quality, so cascades need real held-out evaluation of *both* metrics before shipping, exactly as `example.py` §2 does on a toy setup.

## 4. Putting it together

None of these ideas are mutually exclusive — a production system typically runs all of them at once: a quantized, possibly distilled cheap model handles the easy end of a routing cascade; the expensive tier reuses prefix-cached KV blocks for its shared system prompt; and both tiers batch concurrent requests using a continuous-batching server built on paged KV-cache memory. Cost and latency optimization at the level of a real deployment is the combination of every technique in this phase, applied together against a real traffic distribution, not a single trick applied in isolation.

## Video Script Outline

1. Motivation — every earlier lesson attacked the model or the server; this lesson attacks the system and the traffic
2. The batching trade-off, formalized: fill-wait grows with batch size, throughput ceiling grows with batch size, no batch size wins both
3. Walkthrough of `example.py` §1 — a real simulated request stream, throughput and latency measured across a batch-size sweep
4. Prefix caching: reusing a shared prompt's KV cache, tied back to Lesson 2's KV cache and Lesson 3's PagedAttention
5. Model routing and cascades: cheap classifier or self-confidence, escalate only the hard fraction
6. Walkthrough of `example.py` §2 — a trained toy difficulty classifier, cascade vs. always-cheap vs. always-expensive, real cost/accuracy numbers
7. Why leaky cascades are dangerous: cost savings are visible, accuracy loss on misrouted hard queries is not
8. Recap of the whole phase's toolkit, and a look ahead to [Phase 10](../../Phase-10-Advanced-and-Frontier-Topics/README.md)'s frontier topics

## Further Reading

- Chen, Zaharia, Zou (2023), *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*
- Pope et al. (2022), *Efficiently Scaling Transformer Inference* (the throughput/latency trade-offs of batched serving at scale)
- Provider prompt-caching documentation (Anthropic, OpenAI, and other hosted LLM API providers) for how prefix caching is exposed in production APIs

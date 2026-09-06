# VLM Inference and Deployment

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `10-VLM-Inference-and-Deployment`

## Why this matters

[Phase 09](../../Phase-09-Deployment-and-Inference-Optimization/README.md) built the full picture of LLM inference: prefill versus decode, the KV cache, quantization, continuous batching, speculative decoding, and the roofline arithmetic that says which of them will help. Almost all of it applies to VLMs unchanged — the language model is still the language model. What changes is the **shape of the workload**, and it changes enough to invalidate the intuitions Phase 09 leaves you with. A text chat turn is a hundred-token prompt and a few hundred output tokens, so decode dominates. A VLM request carries 576 or 3,136 vision tokens ([Lesson 1](../01-Vision-Encoders-and-Image-Tokenization/README.md)) before the user's question starts, which makes it prefill-heavy, memory-hungry, and disruptive to batching in ways a text-only serving stack is not tuned for. This lesson measures where the time actually goes and works through the four optimizations that matter specifically because the input is an image.

## What this lesson covers

- Measured: the vision tower, prefill, and KV-cached decode, timed separately
- Why the image's entire cost lands in time-to-first-token, and when a user notices
- Production arithmetic: TTFT and KV-cache footprint across real vision-token budgets
- KV cache as a concurrency limit, and the second reason to compress vision tokens
- Prefix caching: the biggest single win in VLM serving, and what forfeits it
- Chunked prefill, and why one user's image stutters everyone else's stream
- What transfers unchanged from Phase 09, and what doesn't

## 1. Measured: three costs, one of them new

`example.py` §1 builds a small vision tower and a small decoder with a real KV cache and times each phase separately (numbers from one CPU run; the ratios are the point, not the absolute values):

```
vision tokens produced           : 196
vision tower forward             :     39.5 ms
LLM prefill, image + text        :     38.7 ms
LLM prefill, text only           :     12.5 ms   <- the same prompt, no image
decode of 32 tokens (KV-cached)  :    182.3 ms  (5.70 ms/token)

TIME TO FIRST TOKEN              :     78.2 ms   (vision 51%, prefill 49%)
```

Adding one image made prefill 3.1× more expensive, and the vision tower had to run before any of it. The structural point is **where** that cost lands: entirely in time-to-first-token. Per-token decode speed — the tokens/second number a text-LLM benchmark reports — is untouched, because once the KV cache exists each new token costs exactly what it would have in a text-only model.

So an image never slows the stream; it delays the start of it, and it consumes cache and compute that would have served other requests.

## 2. Production arithmetic

Scaling the same structure to a 7B-class VLM on one A100-ish GPU (150 TFLOP/s achieved for prefill, 1.5 TB/s for bandwidth-bound decode, 200 output tokens):

```
configuration                       vis tok   TTFT ms   KV cache MB   total s
text only, no image                       0         9            52      1.88
1 image, 32 resampler queries            32        12            69      1.88
1 image, 144 tok (pixel-shuffle)        144        23           128      1.89
1 image, CLIP@336 = 576 tok             576        63           354      1.93
1 image, AnyRes 896px = 3136 tok      3,136       302         1,697      2.17
4 images @ 576 tok                    2,304       224         1,260      2.09
16 video frames @ 64 tok              1,024       105           589      1.97
64 video frames @ 64 tok              4,096       392         2,200      2.26
```

TTFT rises 30× from the text-only row to a single high-resolution image: as far as cost is concerned, **the vision tokens are the prompt**. Whether that matters to a user depends entirely on output length:

```
 output tokens  text only (s)   AnyRes image (s)   image share
             1           0.02               0.31           94%
            20           0.20               0.49           60%
           200           1.88               2.17           13%
          1000           9.34               9.64            3%
```

For a one-token answer — a yes/no, a class label, a routing decision, a moderation verdict — the image *is* the request. For a long description it is a rounding error on latency. Which means the same model, on the same hardware, has a completely different optimization problem depending on the product it is serving.

The KV column should be read as a **capacity** limit rather than a latency one. An AnyRes image costs about 1.7 GB of KV cache for one request. That number, not FLOPs, is what caps how many concurrent requests a GPU holds; fewer concurrent requests means smaller batches, which means worse throughput per GPU. This is the second, less obvious reason [Lesson 4](../04-Connectors-and-Visual-Token-Compression/README.md)'s token compression pays off: it reduces the per-request latency *and* multiplies the number of requests a GPU can serve at once.

## 3. Prefix caching

The single largest win available in VLM serving, and it applies to an extremely common access pattern — several questions about the same image, or a multi-turn conversation about one document page:

```
 turns about one image   no caching (ms)   prefix cached (ms)   saving
                     1                63                   63        1.0x
                     4               252                   91        2.8x
                    16             1,009                  203        5.0x
```

The vision prefill is paid once instead of once per turn. Two conditions must hold, and they are exactly why some designs give this up:

1. **The vision tokens must be an unchanged prefix.** True for prefix/projector fusion ([Lesson 3](../03-VLM-Architectures-and-Fusion-Strategies/README.md)) with image-first prompts. A system prompt that varies per request, or text spliced before the image, breaks the shared prefix and forfeits the saving.
2. **The vision tokens must not depend on the question.** [Instruction-aware compression](../04-Connectors-and-Visual-Token-Compression/README.md#3-measured-accuracy-vs-compression) buys accuracy per token and gives this up entirely, because its tokens change with every new question. That is a real architectural trade with a serving-cost side, and it is easy to miss when choosing a connector on benchmark scores alone.

Worth adding alongside it: a plain **image-embedding cache** keyed on the image hash, storing the vision tower's output. In an application that answers many questions about a fixed catalogue of images, that converts the tower from a per-request cost into a one-time cost per image.

## 4. Batching, and why images make scheduling harder

```
request                        prefill tokens   chunks of 2048
text chat turn                            100                1
1 image @ 576 + prompt                    676                1
1 AnyRes image @ 3136                   3,236                2
8-image comparison @ 576                4,708                3
64-frame video @ 64                     4,196                3
```

Two problems, both specific to vision-heavy inputs:

**A long prefill blocks the batch.** While the scheduler grinds through 4,000+ vision tokens for one request, every other request in the batch stops producing tokens — one user's image is felt as a stutter by everyone else. **Chunked prefill** (Phase 09 Lesson 4's territory, but far more load-bearing here) splits the prefill into fixed-size pieces and interleaves decode steps between them. For a text-only server it is a nice-to-have; for a VLM server it is close to mandatory.

**Request cost becomes wildly variable.** A text turn and a video request differ by ~40× in prefill work. A scheduler that treats requests as interchangeable will mispredict both latency and memory, so production VLM serving admits requests against a **token budget that counts vision tokens**, not a request count. It is also why VLM endpoints price per image (or per image tile) rather than folding images into the text-token price.

## 5. What transfers from Phase 09, and what doesn't

| Technique | Applies to a VLM? |
|---|---|
| **Quantization** ([Lesson 2](../../Phase-09-Deployment-and-Inference-Optimization/02-Quantization/README.md)) | Yes, to the language model exactly as before. The vision tower is a small fraction of the weights and is often kept at higher precision — it runs once per request, so its precision is cheap, and it is the part whose errors are unrecoverable ([Lesson 7](../07-VLM-Hallucination-and-Alignment/README.md)). |
| **Paged KV cache** | Yes, and more important than in text serving, because sequence lengths vary by an order of magnitude between requests. |
| **Prefix caching** | Yes, and the single biggest win — see §3. |
| **Speculative decoding** ([Lesson 3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md)) | Only helps the decode phase. It does nothing for the vision-dominated prefill, so its benefit on a VLM workload is smaller than the text-only numbers suggest — and nil for short-output requests. |
| **Continuous batching** | Yes, but requires chunked prefill and token-budget admission to work well — see §4. |
| **Chunked prefill** | Effectively mandatory rather than optional. |
| **Distillation / pruning** ([Lesson 5](../../Phase-09-Deployment-and-Inference-Optimization/05-Model-Distillation-and-Pruning/README.md)) | Yes, and for VLMs "distillation" often means training a smaller model on a larger VLM's outputs — which inherits its hallucinations too (Lesson 6 §6). |
| **Token compression** | The VLM-specific optimization with no text analogue: it cuts prefill FLOPs, KV memory, and concurrency limits at once. |

The summary a deployment engineer should leave with: **profile TTFT and KV footprint, not tokens per second.** Every VLM-specific lever — resolution, tiling, connector choice, prefix caching, chunked prefill — moves those two numbers, and none of them show up in a tokens-per-second measurement.

## Video Script Outline

1. Why Phase 09's intuitions need adjusting: the workload shape, not the model, is what changed
2. The measured breakdown: vision tower, prefill with and without the image, KV-cached decode
3. Where the cost lands — TTFT, never the stream — and the output-length table that says when it matters
4. The production table: 30× TTFT from one high-resolution image
5. KV cache as the concurrency limit, and compression's double payoff
6. Prefix caching, and the two conditions that make it possible
7. The connector choice with a serving-cost consequence: instruction-aware compression forfeits caching
8. Chunked prefill: one user's image stuttering everyone else's stream
9. Token-budget admission and why images are priced separately
10. The transfer table from Phase 09 — including where speculative decoding disappoints
11. Recap: profile TTFT and KV footprint, not tokens/second
12. Preview of [Lesson 11](../11-Beyond-Vision-Full-Multimodality/README.md)

## Further Reading

- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM; paged KV cache and prefix sharing)
- Agrawal et al. (2024), *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (chunked prefill, the mechanism §4 depends on)
- Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention prefix caching across requests)
- Wang et al. (2024), *Qwen2-VL* (native dynamic resolution and frame merging, i.e. token budget as a first-class design parameter)
- Phase 09 Lessons [3](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md), [4](../../Phase-09-Deployment-and-Inference-Optimization/04-Serving-Frameworks/README.md) and [10](../../Phase-09-Deployment-and-Inference-Optimization/10-Production-Serving-and-Benchmarking/README.md) — the text-only groundwork this lesson modifies

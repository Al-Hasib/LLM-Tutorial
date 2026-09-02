# Survey of Popular Open LLMs

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `07-Survey-of-Popular-Open-LLMs`

## Why this matters

Every concept in this phase — decoder-only architecture, Mixture of Experts, RoPE, sliding-window attention — is not academic trivia; it's the literal parts list of the open LLMs you can download and run today. This lesson is a guided tour that maps each real, widely-used open model back to the specific lessons in this course that explain how it works, plus one genuinely new, practically important technique this survey motivates: **Grouped-Query Attention (GQA)**, a variant of multi-head attention designed specifically to make inference cheaper.

## What this lesson covers

- LLaMA (1/2/3): the modern open-model architecture baseline
- Mistral and Mixtral: sliding-window attention and Mixture of Experts in production
- Other notable families: Falcon, Qwen, Gemma, Phi, DeepSeek
- Grouped-Query Attention (GQA): the multi-head/multi-query middle ground
- Reading a model's config file with fresh eyes

## 1. LLaMA: the modern open baseline

Meta's LLaMA family (Touvron et al., 2023) established the architectural recipe most subsequent open decoder-only models converge on:

- **RoPE** for positional encoding ([Lesson 6 §1](../06-Long-Context-Techniques/README.md#1-rope-rotating-q-and-k-instead-of-adding-to-the-input))
- **RMSNorm** instead of LayerNorm ([Phase 02 Lesson 5 §2](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#2-layer-normalization)) — a simplified normalization that skips re-centering (no mean subtraction, only rescaling by root-mean-square), cheaper to compute with little quality loss
- **SwiGLU** activation in the FFN instead of GELU ([Phase 02 Lesson 5 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#5-the-position-wise-feed-forward-network-ffn)) — a gated variant that adds a learned multiplicative gate to the FFN's hidden layer
- **Pre-LN** residual structure ([Phase 02 Lesson 5 §4](../../Phase-02-Transformer-Architecture-Deep-Dive/05-LayerNorm-Residuals-FFN/README.md#4-pre-ln-vs-post-ln))
- Trained well past Chinchilla-optimal token counts for cheaper inference ([Lesson 5 §6](../05-Scaling-Laws/README.md#6-practical-implications))

LLaMA is, architecturally, almost exactly [Phase 02 Lesson 6's mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) — this specific set of small, well-tested substitutions, at a much larger scale.

## 2. Mistral and Mixtral

Mistral 7B (Jiang et al., 2023) added **sliding-window attention** ([Lesson 6 §3](../06-Long-Context-Techniques/README.md#3-sliding-window-local-attention-fixing-the-compute-not-just-position)) on top of the LLaMA-style recipe, plus **Grouped-Query Attention** (introduced fully below) to reduce inference memory cost. Mixtral (Jiang et al., 2024) takes the same base architecture and replaces the dense FFN with a **Mixture of Experts** layer ([Lesson 4](../04-Mixture-of-Experts/README.md)) — 8 experts, top-2 routing — giving it a much larger total parameter count than a dense model with the same per-token compute cost.

## 3. Other notable families

| Model family | Notable architectural choices |
|---|---|
| **Falcon** | LLaMA-style backbone; an early large-scale open release with a carefully filtered web-scale dataset |
| **Qwen** | LLaMA-style backbone with strong multilingual tokenizer coverage; larger releases use MoE |
| **Gemma** | LLaMA-style backbone with some normalization placement tweaks; distilled from larger internal models |
| **Phi** | Comparatively small models emphasizing very high-quality, curated training data over sheer scale |
| **DeepSeek-V2/V3** | Aggressive Mixture of Experts (many small experts, fine-grained routing) combined with long-context techniques |

The pattern across nearly all of them: **the same small set of architectural building blocks from this course, recombined and re-tuned**, not fundamentally new architectures each time.

## 4. Grouped-Query Attention (GQA): a new, practically important variant

Multi-head attention ([Phase 02 Lesson 2](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md)) gives every head its *own* `K` and `V` projections. At inference time, autoregressive generation caches every previous token's `K` and `V` vectors (the "KV cache," covered fully in [Phase 09: KV Cache and Speculative Decoding](../../Phase-09-Deployment-and-Inference-Optimization/03-KV-Cache-and-Speculative-Decoding/README.md)) so they don't need to be recomputed at every new generation step — and that cache's size scales directly with the number of separate K/V projections a model has.

Two variants trade quality for cache size:

- **Multi-Query Attention (MQA)**: every head shares **one single** `K`/`V` projection (only `Q` stays per-head) — dramatically smaller KV cache, but a noticeable quality cost from that much reduced representational diversity.
- **Grouped-Query Attention (GQA)**: the middle ground. Split heads into `g` groups; heads within a group **share** one `K`/`V` projection, but different groups still get their own. `g=1` recovers MQA exactly; `g=num_heads` recovers ordinary multi-head attention exactly.

```
MHA:  num_heads separate K/V projections   (best quality, largest KV cache)
GQA:  g groups, g separate K/V projections  (tunable middle ground)
MQA:  1 shared K/V projection for all heads (smallest KV cache, most quality loss)
```

LLaMA 2 70B, LLaMA 3, and Mistral all use GQA specifically to make serving long-context conversations cheaper without paying MQA's full quality cost.

## Video Script Outline

1. Motivation — "every concept in this phase, mapped onto models you can actually download"
2. LLaMA: RoPE + RMSNorm + SwiGLU + Pre-LN + trained-past-Chinchilla, tie each to its lesson
3. Mistral (sliding-window) and Mixtral (MoE), tie to Lessons 4 and 6
4. Quick tour: Falcon, Qwen, Gemma, Phi, DeepSeek — what's genuinely different, what isn't
5. Grouped-Query Attention: the KV-cache-size motivation, MHA/GQA/MQA as one spectrum
6. Walkthrough of `example.py` — implement GQA, measure exact KV-cache size across MHA/GQA/MQA, and build a small architecture-comparison table across the surveyed models
7. Recap: three architecture families, MoE, scaling laws, long-context, and now GQA -> this phase's full toolkit, ready for Phase 04's pretraining

## Further Reading

- Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models* and *Llama 2: Open Foundation and Fine-Tuned Chat Models*
- Jiang et al. (2023), *Mistral 7B*; Jiang et al. (2024), *Mixtral of Experts*
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)

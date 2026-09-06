# VLM Architectures and Fusion Strategies

**Phase:** [Vision-Language Models](../README.md) · **Topic folder:** `03-VLM-Architectures-and-Fusion-Strategies`

## Why this matters

[Lesson 1](../01-Vision-Encoders-and-Image-Tokenization/README.md) produced a sequence of vision vectors and [Lesson 2](../02-Vision-Language-Pretraining-Objectives/README.md) decided what those vectors mean. This lesson answers the question everything so far has deferred: **where do they enter the language model?** There are exactly three answers in current practice — concatenate them onto the text sequence, let the text cross-attend to them, or refuse to separate the two modalities in the first place — and the choice is not cosmetic. It fixes how much of the LLM you have to retrain, how much context each image consumes, how inference cost scales with image count, and whether the model's text-only ability survives the operation at all. Every architectural diagram in a VLM paper is, at bottom, a choice among these three.

## What this lesson covers

- Why the naive option (pool the image into one vector) fails, measured on a grounding task
- **A. Prefix / projector fusion** (LLaVA and most open VLMs): images become tokens
- **B. Cross-attention fusion** (Flamingo, Idefics, Llama-3-V): images stay outside the sequence
- **C. Early / native fusion** (Chameleon, Fuyu): no separate tower at all
- The zero-initialized `tanh` gate, and why a frozen LLM survives having new layers inserted
- The cost asymmetry that decides the choice in production: quadratic sequence growth vs. linear cross-attention
- Interleaving, multi-image, and where each strategy breaks

## 1. The task, and why pooling fails

`example.py` builds a miniature grounding problem: a scene of six objects, each handed to the model as one vision token encoding both a shape and a colour; the question names one shape and asks for its colour. Chance is 16.7%. The measured results with everything trainable:

```
architecture                        accuracy     params  text seq len  attn cells
blind (text only, no image)           16.7%    155,270             4          16
pooled image -> 1 token               43.2%    161,542             5          25
A. prefix / projector (LLaVA)        100.0%    161,542            10         100
B. cross-attention (Flamingo)        100.0%    307,724             4          16
C. early / native fusion             100.0%    157,510            10         100
```

The blind model is pinned at chance, as it must be. The interesting row is the pooled one: averaging the vision tokens into a single vector — precisely what a CLIP embedding is — gets 43.2%, well above chance and nowhere near solving the task. The averaged vector still carries *which* colours are present, so guessing among those beats guessing among all six; what it has lost is which colour is **bound** to which shape. This is [Lesson 1 §6](../01-Vision-Encoders-and-Image-Tokenization/README.md#6-pooling-one-vector-or-all-of-them)'s pooling argument restated as task accuracy, and it is the reason every VLM in this lesson keeps a *sequence* of vision tokens. All three real fusion strategies solve the task outright.

## 2. A. Prefix / projector fusion

The dominant design, and the one [Phase 10 Lesson 1 §3](../../Phase-10-Advanced-and-Frontier-Topics/01-Multimodal-LLMs/README.md#3-from-an-aligned-space-to-a-multimodal-llm-llava) sketched:

```
vision tokens --[projector]--> LLM embedding space
sequence = [ vis₁ … vis_N | txt₁ … txt_M ]        # one flat sequence
```

Nothing in the LLM changes. The vision tokens are just embeddings, and ordinary causal self-attention lets any text token attend back to any of them. Its advantages are almost entirely practical: the new parameter count is tiny (an MLP), any existing LLM inference stack serves it without modification, and interleaving images and text anywhere in a conversation is trivially expressible — you simply splice vision tokens into the sequence at the right position.

Its cost is that images now live in the context window. Self-attention is quadratic in total sequence length, every vision token gets a KV-cache entry in every layer, and a 576- or 3,136-token image (Lesson 1's real numbers) can dwarf the user's actual question. In the toy run above, prefix fusion turned a 4-token sequence into a 10-token one, taking attention from 16 cells to 100.

## 3. B. Cross-attention fusion

Flamingo's design, revived in Idefics and Llama-3-V: leave the text stream alone, and insert new **gated cross-attention** blocks between the LLM's existing layers, where text queries attend to vision keys/values.

```
for each layer:
    text = text + tanh(gate) · CrossAttn(q=text, kv=vision)
    text = LLM_layer(text)                        # unchanged, frozen
```

The text sequence never grows. The image is paid for in cross-attention, whose cost is `text_len × vision_len` — **linear** in the number of vision tokens instead of quadratic in their addition to the sequence. For many images, or very high-resolution ones, this is a structurally better deal, which is why cross-attention keeps returning in models targeting long interleaved documents and video.

The prices are real too: it adds substantial new parameters (in the toy model, ~50% of the total, since a whole gated block joins every layer), it requires modifying the model's forward pass — so a stock text-only inference server will not run it — and expressing "image, then text, then a second image" requires explicit masking machinery to control which text spans may see which image, rather than falling out of sequence order for free.

### The zero-initialized gate

The `tanh(gate)` wrapper with `gate` initialized to **zero** is the detail that makes the whole approach viable on a frozen LLM. At initialization `tanh(0) = 0`, so every inserted block is an exact identity and the modified model computes bit-for-bit what the original computed. `example.py` §3 checks this by loading the same language-model weights into a cross-attention model and a text-only model and comparing outputs:

```
max |cross-attn model - text-only model| at init : 0.00e+00
gate values after training, layer by layer      : +0.125, -0.134, -0.133
```

Training then opens the gates only as far as the vision signal is worth, and the resulting values are a directly readable measure of how much image each layer let in. The same zero-init trick appears in [LoRA](../../Phase-05-Finetuning-LLMs/02-LoRA-and-QLoRA/README.md) (the `B` matrix is zero-initialized so the adapter starts as a no-op) and in ControlNet — it is the standard way to add capacity to a working model without breaking it on step one.

## 4. C. Early / native fusion

Chameleon and Fuyu take the position that a separate pretrained vision tower is itself the problem. In early fusion there is no tower interface at all: raw patch vectors (Fuyu applies one linear layer to raw pixels; Chameleon quantizes images into discrete tokens with a VQ tokenizer) enter the same Transformer as text from the first layer of pretraining, and one set of weights processes both modalities throughout.

The advantages are conceptual and, at scale, real: no frozen tower means no ceiling imposed by what the tower discarded (Lesson 2 §4), no resolution constraint inherited from someone else's pretraining, and — for Chameleon — image *generation* falls out of the same next-token objective as text, because images are just tokens in the vocabulary ([Lesson 11](../11-Beyond-Vision-Full-Multimodality/README.md) returns to this).

The disadvantage is cost. There is no pretrained language model to freeze because the text and vision weights were never separate; you pay for a full multimodal pretraining run, and early multimodal training is notoriously unstable (Chameleon's paper devotes real space to the norm-growth and divergence problems it had to solve). `example.py` includes early fusion in the frozen-backbone table for completeness, but that column does not really apply to it — the toy task is simply easy enough that a frozen backbone still solves it.

## 5. The frozen-LLM view

A real VLM starts from a language model that already works, so "how much new machinery does this strategy need?" is often the deciding question. With the LLM frozen:

```
architecture                        accuracy   trainable  % of model
A. prefix / projector (LLaVA)        100.0%       6,662       4.1%
B. cross-attention (Flamingo)        100.0%     152,844      49.7%
C. early / native fusion              99.7%       2,630       1.7%
```

Prefix fusion trains a projector and nothing else — that ratio is the single biggest reason LLaVA-style recipes dominate open-source VLM work: you can build a credible VLM on one GPU in hours. Cross-attention adds a lot of parameters but touches no existing weight, which is exactly what let Flamingo attach vision to a frozen 70B model when full fine-tuning of that model was out of reach.

## 6. Choosing

| | Prefix / projector | Cross-attention | Early / native |
|---|---|---|---|
| New parameters | tiny (MLP) | large (block per layer) | none separate |
| LLM weights touched | none required | none | all (trained jointly) |
| Image cost in LLM | quadratic (context growth) | linear (cross-attn) | quadratic |
| Serving stack | any stock LLM server | needs custom forward | custom |
| Interleaving / multi-image | free (sequence order) | needs masking machinery | free |
| Image generation | no | no | yes (Chameleon-style) |
| Typical users | LLaVA, Qwen-VL, InternVL, most open VLMs | Flamingo, Idefics, Llama-3-V | Chameleon, Fuyu |

The practical rule: prefix fusion unless the token budget forces your hand. When images dominate the context — many images per request, long video, or high-resolution documents — cross-attention's linear scaling stops being a nicety, and the alternative is compressing the vision tokens instead, which is [Lesson 4](../04-Connectors-and-Visual-Token-Compression/README.md).

## Video Script Outline

1. The task: six objects, "what colour is the triangle?" — grounding in miniature
2. The blind and pooled baselines: 16.7% and 43.2%, and what the pooled model's partial credit reveals about binding
3. A. Prefix fusion — images become tokens; nothing in the LLM changes; the context bill arrives later
4. B. Cross-attention — the text stream stays short; the image is paid for linearly
5. The zero-init gate, demonstrated: identical outputs at init, non-zero gates after training
6. C. Early fusion — one Transformer, no tower, no frozen anything, image generation for free
7. The frozen table: 4.1% vs 49.7% trainable, and what each buys
8. The decision table, and the case where prefix fusion stops being the obvious answer
9. Recap + preview of [Lesson 4](../04-Connectors-and-Visual-Token-Compression/README.md): shrinking the vision sequence itself

## Further Reading

- Liu, Li, Wu, Lee (2023), *Visual Instruction Tuning* (LLaVA; prefix/projector fusion)
- Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning* (gated cross-attention and the zero-init gate)
- Laurençon et al. (2024), *What matters when building vision-language models?* (a controlled comparison of these exact choices; the Idefics2 ablations)
- Team Chameleon (2024), *Chameleon: Mixed-Modal Early-Fusion Foundation Models*
- Bavishi et al. (2023), *Fuyu-8B: A Multimodal Architecture for AI Agents* (linear patch projection straight into the decoder)

# LoRA and QLoRA

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `02-LoRA-and-QLoRA`

## Why this matters

[Lesson 1](../01-Full-Finetuning-vs-PEFT/README.md) established the general PEFT strategy — freeze the base model, train a small add-on — and the memory math that motivates it. This lesson makes that add-on concrete with the single most widely used PEFT method in practice: **LoRA**. Nearly every "fine-tune an open LLM on my own data" workflow you'll find today (including [Lesson 5's Hugging Face walkthrough](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md)) defaults to LoRA or its quantized cousin, QLoRA, specifically because of the mechanism this lesson builds from scratch.

## What this lesson covers

- LoRA's core idea: a frozen weight matrix plus a trainable low-rank update
- Why the low-rank update has so few parameters, and how the rank `r` controls the trade-off
- Merging LoRA weights back into the base model at inference time, for zero extra latency
- QLoRA: 4-bit quantization of the frozen base, enabling fine-tuning on far less GPU memory
- Building a LoRA linear layer from scratch in PyTorch, and verifying the frozen weights truly never change

## 1. The core idea: freeze `W`, learn a low-rank update

Hu et al. (2021) observed that the *change* a fine-tuning run makes to a pretrained weight matrix tends to have a much lower "intrinsic rank" than the matrix itself — the update doesn't need the full expressive power of a dense `d x k` matrix to be useful. LoRA exploits this directly: instead of fine-tuning a weight matrix `W` (shape `d x k`), **freeze `W` entirely** and add a trainable update expressed as the product of two much smaller matrices:

```
W' = W + (alpha / r) * B @ A

W : d x k    (frozen -- the original pretrained weight, never updated)
A : r x k    (trainable, initialized to small random values)
B : d x r    (trainable, initialized to all zeros)
r << min(d, k)                (the LoRA "rank" -- typically 4-64)
alpha                          (a fixed scaling constant; alpha/r sets the update's magnitude)
```

`B` is initialized to zero so that at the very start of training, `B @ A = 0` and `W' = W` exactly — fine-tuning starts from the pretrained model's exact original behavior and only gradually diverges as `A` and `B` learn. The forward pass for an input `x` becomes:

```
h = x @ W.T + (alpha / r) * x @ A.T @ B.T
```

Only `A` and `B` receive gradients; `W` sees gradients flow *through* it during backpropagation (the chain rule still needs `dL/dx`) but never gets a gradient *update*, exactly the frozen-parameter mechanic from [Lesson 1 §4](../01-Full-Finetuning-vs-PEFT/README.md#4-the-peft-idea-freeze-almost-everything).

## 2. Why the parameter count drops so much

A dense `d x k` weight matrix has `d * k` parameters. LoRA's update matrices have `r * k + d * r = r * (d + k)` parameters instead. For a `4096 x 4096` attention projection matrix (a realistic size for a 7B-parameter-class model) and `r = 16`:

```
dense:  4096 * 4096          = 16,777,216 parameters
LoRA:   16 * (4096 + 4096)   =    131,072 parameters   (~0.78% of the dense count)
```

`example.py` computes this exactly, at several values of `r`, and confirms it against real PyTorch parameter counts.

## 3. Merging at inference time: zero extra latency

Because `W` and `B @ A` are both just `d x k` matrices, they can be **added together once**, offline, after training finishes:

```
W_merged = W + (alpha / r) * B @ A
```

`W_merged` is a single ordinary weight matrix, identical in shape to the original. Deploying it costs *exactly* the same inference-time compute and latency as the original unmodified model — there is no extra matrix multiplication at serving time, unlike adapters or prefix tuning ([Lesson 3](../03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md)), which add a small amount of compute to every forward pass. This "merge for free" property is one of LoRA's biggest practical advantages, and it's also reversible: keep `A` and `B` around separately, and you can swap between many merged variants of the same base model without ever re-loading the frozen weights.

## 4. QLoRA: LoRA on a quantized base

Dettmers et al. (2023) pushed LoRA's memory savings further with **QLoRA**: since the base weights `W` are frozen and never updated, they don't need to be stored at full precision at all — QLoRA stores them in a specially designed **4-bit format (NF4, "4-bit NormalFloat")**, optimized for the roughly-Gaussian distribution that pretrained neural network weights tend to follow, and *dequantizes* them back to a higher precision (e.g., bf16) on the fly, block by block, only for the duration of each forward/backward computation. `A` and `B` (the small trainable pieces) are still kept and updated in full precision, since they're the part that actually needs to learn. Three techniques combine to make this work well:

- **NF4 quantization** — a 4-bit format whose quantization levels are placed to match the expected distribution of pretrained weights, giving lower quantization error than a naive uniform 4-bit scheme for this specific use case.
- **Double quantization** — the small per-block scaling constants that quantization itself requires are *themselves* quantized, shrinking their overhead further.
- **Paged optimizers** — GPU memory pages for optimizer state spill to CPU memory automatically when a batch causes a memory spike, avoiding out-of-memory crashes without manual intervention.

The combined effect: QLoRA reduced the GPU memory needed to fine-tune a 65B-parameter model from well over 780 GB (full fine-tuning) to under 48 GB — small enough for a single high-end consumer/workstation GPU — while matching full fine-tuning's task performance in the paper's benchmarks. `example.py` implements a simplified from-scratch simulation of 4-bit block quantization to make the mechanism (and its error/memory trade-off) concrete.

## 5. What this lesson's code does (and what a real workflow uses instead)

`example.py` implements a `LoRALinear` module completely from scratch in raw PyTorch — no external PEFT library — because seeing the mechanism explicitly is the point of this lesson. A real project would instead use Hugging Face's `peft` library (`LoraConfig`, `get_peft_model`) to apply exactly this mechanism to a real pretrained model in a few lines, which is exactly what happens "under the hood" — [Lesson 5](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) walks through that real API and maps every config option directly back to the `LoRALinear` class built here.

## Video Script Outline

1. Motivation — "the single most common way real LLMs get fine-tuned today, built from scratch"
2. The frozen-`W`-plus-low-rank-update idea, and why zero-initializing `B` matters
3. Parameter-count math: dense `d*k` vs. LoRA's `r*(d+k)`, made concrete with real numbers
4. Merging back into `W` for zero-extra-latency inference
5. QLoRA: 4-bit NF4 quantization of the frozen base, double quantization, paged optimizers
6. Walkthrough of `example.py` — train a LoRA layer, verify `W` truly never changes, compare parameter counts across ranks, simulate 4-bit quantization error/memory trade-offs
7. Recap + preview: Lesson 3 covers the PEFT alternatives that don't merge as cleanly (prompt/prefix tuning, adapters)

## Further Reading

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*
- Dettmers et al. (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (an earlier quantization technique that QLoRA builds on conceptually)
- Hugging Face `peft` library documentation, `LoraConfig` and `get_peft_model` (the real-world API — see [Lesson 5](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md))

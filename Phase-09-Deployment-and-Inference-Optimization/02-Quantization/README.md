# Quantization

**Phase:** [Deployment and Inference Optimization](../README.md) · **Topic folder:** `02-Quantization`

## Why this matters

Every model in this course so far has been trained and stored in 32-bit (or 16-bit) floating point. That's the right choice for training — gradients need precision to accumulate correctly — but it's expensive for *serving*: a 7-billion-parameter model in float32 needs 28 GB just to hold its weights, before a single token of context is processed. Quantization is the first and most impactful lever for closing that gap: it shrinks weights (and sometimes activations) down to 8-bit or 4-bit integers, cutting memory footprint and often latency by 2-8x, in exchange for a small, carefully-controlled amount of numerical error.

This lesson is the natural next stop after [Lesson 1](../01-GPU-and-Hardware-Fundamentals/README.md) because it directly exploits the memory-bandwidth-bound reality that lesson establishes: on memory-bandwidth-bound hardware, a smaller weight means less to move per step, not just less to store. Everything downstream compounds with it too: the KV cache covered in [Lesson 3](../03-KV-Cache-and-Speculative-Decoding/README.md) is frequently stored in reduced precision too; the GQA/MQA cache-size discussion from [Phase 03 Lesson 7](../../Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md#4-grouped-query-attention-gqa-a-new-practically-important-variant) and quantization are complementary, independent levers on the same memory bottleneck; and the serving frameworks in [Lesson 4](../04-Serving-Frameworks/README.md) (particularly llama.cpp) are built specifically around running quantized model formats efficiently. It also pairs naturally with [Lesson 5](../05-Model-Distillation-and-Pruning/README.md)'s distillation and pruning — all three are "make the model smaller/cheaper without retraining from scratch" techniques, tackled from different angles — and directly feeds the cost/latency trade-offs of [Lesson 6](../06-Cost-and-Latency-Optimization/README.md).

## What this lesson covers

- The core trade-off: model size and speed vs. numerical accuracy
- INT8/INT4 quantization mechanics: scale, zero-point, quantize/dequantize
- Symmetric vs. asymmetric quantization
- GPTQ: layer-by-layer, Hessian-aware post-training quantization
- AWQ: protecting activation-salient weights instead of using second-order math
- Why not every weight matters equally to a quantized layer's output

## 1. The core trade-off

A model's weights are just tensors of numbers. Storing each number in fewer bits directly shrinks:

- **Memory footprint** — how much RAM/VRAM is needed just to hold the model (and thus how many concurrent requests fit, or whether the model fits on a given device at all)
- **Memory bandwidth cost** — on modern hardware, moving weights from memory to the compute units is frequently the actual bottleneck (not the arithmetic itself, [Lesson 1](../01-GPU-and-Hardware-Fundamentals/README.md)'s memory-bound regime), so smaller weights often means *faster* inference too, not just less storage
- **Numerical accuracy** — the weight's exact original value is lost; only an approximation survives

The engineering problem quantization research solves is: **how do you throw away the most bits while losing the least accuracy?** Naively rounding every weight independently to the nearest representable low-bit value ("round-to-nearest," RTN) is the simplest approach and works surprisingly well down to INT8, but degrades badly at INT4 and below — which is why GPTQ and AWQ (below) exist.

## 2. INT8/INT4 quantization mechanics

The standard scheme is **uniform affine quantization**. For a tensor of float32 weights `x`, symmetric quantization (no zero-point, assumes values are roughly centered around zero — a good fit for neural network weights) works as:

```
scale = max(|x|) / (2^(bits-1) - 1)

quantize:    q = round(x / scale)              # q is an integer in [-(2^(bits-1)-1), 2^(bits-1)-1]
dequantize:  x_hat = q * scale                  # x_hat approximates x, in float
```

For `bits=8`, `q` ranges over roughly [-127, 127]; for `bits=4`, just [-7, 7] — only 15 distinct integer values to represent an entire weight matrix's worth of numbers, which is why INT4 is so much more error-prone than INT8.

**Asymmetric quantization** adds a **zero-point** `z` and is useful when values are not centered around zero (common for post-activation values, which after a ReLU are all non-negative):

```
scale = (max(x) - min(x)) / (2^bits - 1)
z     = round(-min(x) / scale)

quantize:    q = round(x / scale) + z
dequantize:  x_hat = (q - z) * scale
```

`example.py` implements the symmetric scheme from scratch on random weight matrices at both 8-bit and 4-bit, and directly measures the reconstruction error (mean squared error and max absolute error between the original weights and the dequantized approximation) alongside the actual memory savings.

## 3. GPTQ: Hessian-aware post-training quantization

Round-to-nearest quantizes each weight independently, ignoring how weights interact. **GPTQ** (Frantar et al., 2022) improves on this: it quantizes a layer's weight matrix **column by column**, and after quantizing each column, it adjusts the *remaining, not-yet-quantized* columns slightly to compensate for the error just introduced — using an approximation to the layer's **Hessian** (second-order curvature information derived from a small calibration dataset's activations) to figure out which compensating adjustment minimizes the resulting error in the **layer's output**, not just the raw weight values.

The key conceptual shift: GPTQ doesn't ask "is this quantized weight close to the original weight?" — it asks "does this quantized *layer*, run on real calibration data, produce close to the same *output* as the original layer?" Those are different objectives, and optimizing the second one directly is what lets GPTQ push all the way to 4-bit (or even 3-bit) with much smaller quality loss than naive RTN quantization at the same bit-width.

## 4. AWQ: activation-aware weight quantization

**AWQ** (Lin et al., 2023) takes a cheaper, complementary observation: it doesn't need any second-order Hessian math at all. Instead, it observes that when you run real calibration data through a layer, a **small percentage of weight columns** — specifically, those that get multiplied by consistently **large-magnitude activations** — matter disproportionately to the layer's output. Quantizing those particular columns coarsely (to low bits) causes an outsized amount of output error, even though the *weight values themselves* aren't necessarily unusual — it's their interaction with large activations that makes rounding errors there get amplified downstream.

AWQ's fix: identify that small salient fraction of columns from activation statistics (no backward pass or Hessian required — just a forward pass over a calibration set), and either keep those columns at higher precision (a mixed-precision layer) or, in the original paper, per-channel scale them up before quantization and back down after (mathematically pushing more of the limited quantization "resolution" onto the columns that need it) so an ordinary uniform quantizer protects them implicitly. Both mechanisms achieve the same goal: unequal protection for unequal importance.

```
naive RTN INT4:      quantize every weight column uniformly to INT4
AWQ-style INT4:       identify top-s% columns by |activation| magnitude
                      keep those columns at full precision (or scaled favorably)
                      quantize the remaining (1-s)% columns aggressively to INT4
```

`example.py` implements a simplified version of this second mixed-precision strategy directly: a synthetic per-column "activation magnitude" vector stands in for real calibration statistics, and the top-k% columns by that magnitude are kept at fp32 while the rest are quantized to INT4 — then compares reconstruction error against naive uniform INT4 at a matched *effective* average bit-width.

## 5. What quantization does and doesn't cost you

- **What you save**: memory (linear in bits-per-weight), memory bandwidth, and often wall-clock latency, especially on memory-bandwidth-bound hardware
- **What you risk**: accuracy degradation, which grows sharply below INT8 without a smarter scheme like GPTQ or AWQ; certain layers (e.g. the final output projection, or specific "outlier" activation channels documented in the LLM.int8() paper) are more sensitive than others and are sometimes left at higher precision even in an otherwise 4-bit model
- **What doesn't change**: the model's architecture, parameter count, or training — quantization is applied *after* training (post-training quantization, PTQ) and requires no gradient updates to the original model (though it may use a small calibration dataset), which is why it's such a cheap, popular first lever compared to retraining a smaller model from scratch ([Lesson 5](../05-Model-Distillation-and-Pruning/README.md))

## Video Script Outline

1. Motivation — a 7B float32 model needs 28 GB just to sit in memory; quantization is the cheapest way to shrink that
2. The core mechanism: scale, quantize, dequantize — symmetric vs. asymmetric, worked through by hand
3. Why round-to-nearest breaks down at INT4: fewer representable values, bigger rounding error
4. GPTQ: quantize column-by-column, use Hessian information to compensate remaining columns, minimize LAYER OUTPUT error
5. AWQ: skip the Hessian math, just protect the small percentage of activation-salient weight columns
6. Walkthrough of `example.py` — INT8/INT4 quantization from scratch, measured error and memory savings, then the AWQ-style mixed-precision demo
7. Recap: quantization as the first of three "shrink it after training" techniques in this phase, alongside KV-cache tricks and distillation/pruning

## Further Reading

- Frantar, Ashkboos, Hoefler, Alistarh (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- Lin, Tang, Tang, Yang, Dang, Han (2023), *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
- Dettmers, Lewis, Belkada, Zettlemoyer (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*

# Mixed Precision and Optimization

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `04-Mixed-Precision-and-Optimization`

## Why this matters

[Lesson 3](../03-Distributed-Training-Basics/README.md) showed *where* the bytes and FLOPs of a training step get spread across devices. This lesson attacks the size of those bytes and the quality of each optimizer step directly, independent of how many devices you have. Every real pretraining run uses **mixed precision** (not plain fp32, the default in every PyTorch example you've written so far in this course) and **AdamW with a warmup+decay schedule** (not plain SGD). Both choices sound like implementation details, but each is the direct fix for a specific, well-understood failure mode — and both numbers you computed in [Lesson 3's memory calculator](../03-Distributed-Training-Basics/README.md#5-zero--fsdp-shard-dont-replicate) (the `16` bytes/parameter figure) come directly from the precision choices explained here.

## What this lesson covers

- fp32 vs. fp16 vs. bf16: the range/precision trade-off, and why bf16 won for LLM training
- Loss scaling: why fp16 gradients underflow, and how scaling fixes it
- AdamW: momentum + adaptive learning rates, and the weight-decay fix over plain Adam
- Learning-rate schedules: why warmup matters, and why cosine/linear decay follows it

## 1. fp32 vs. fp16 vs. bf16

All three are 4-byte/2-byte binary floating-point formats, differing only in how their bits split between **exponent** (controls *range* — how large or small a number can be) and **mantissa** (controls *precision* — how many significant digits it can represent):

```
fp32 (standard, "full precision"): 1 sign + 8 exponent bits + 23 mantissa bits
fp16 ("half precision"):           1 sign + 5 exponent bits + 10 mantissa bits
bf16 ("bfloat16"):                 1 sign + 8 exponent bits +  7 mantissa bits
```

- **fp32** is the safe default every earlier lesson in this course used — full range, full precision, 4 bytes per value.
- **fp16** halves the memory and roughly doubles throughput on hardware with fp16 tensor cores, but its 5 exponent bits give it a *much* narrower representable range than fp32 (roughly `6e-5` to `65504` in normal range) — values outside that range **overflow to infinity or underflow to zero**. Gradients in deep networks routinely take on very small magnitudes, which makes fp16 underflow a real, frequent problem during training (§2).
- **bf16** keeps fp32's *same 8 exponent bits* (same range as fp32, so no overflow/underflow surprises) but sacrifices precision instead, dropping to only 7 mantissa bits. This trade — same range as fp32, less precision than fp16 — turned out to be exactly the right one for neural network training: overflow/underflow crashes a training run outright, while reduced precision mostly just adds a small amount of numerical noise that gradient descent already tolerates well. This is why bf16 (not fp16) became the standard choice for training modern LLMs on hardware that supports it (TPUs from the start; NVIDIA GPUs from the Ampere generation onward).

## 2. Loss scaling: fp16's underflow problem, and the fix

Consider a real gradient value of `0.00003` (a very ordinary magnitude deep in a large network). fp16's smallest representable *normal* value is around `6e-5` — so `0.00003` rounds straight to **zero** the moment it's cast to fp16, and that entire piece of gradient signal silently vanishes. Do this for enough parameters and training stalls or diverges for no apparent reason.

**Loss scaling** fixes this without changing the math at all: multiply the loss by a large constant `S` (e.g. `2^16 = 65536`) *before* calling `.backward()`. Since gradients scale linearly with the loss they came from, every gradient in the backward pass gets scaled by the same factor `S`, pushing previously-vanishing small values back up into fp16's representable range. After backprop, **unscale** the gradients (divide by `S`) before the optimizer step, recovering the mathematically correct gradient magnitudes:

```
scaled_loss = loss * S
scaled_loss.backward()                  # gradients come out ~S times larger, now representable in fp16
unscaled_grad = fp16_grad.float() / S   # recover the true gradient before the optimizer step
```

`example.py` demonstrates this numerically: casting a small gradient value straight to fp16 underflows it to exactly `0.0`, while scaling first, then unscaling after, recovers the original value almost exactly. (bf16's much wider range means it rarely needs loss scaling at all — another point in its favor, and part of why it simplified mixed-precision training recipes once hardware supported it.)

## 3. AdamW: decoupled weight decay

Recall Adam from earlier optimizer coverage: it keeps a running exponential moving average of each parameter's gradient (`m`, the momentum/first-moment estimate) and of each gradient's *squared* magnitude (`v`, the second-moment estimate), then updates each parameter by its momentum, scaled *inversely* by the square root of its variance estimate — giving every parameter its own adaptive effective learning rate:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)              # bias correction (early steps are biased toward 0)
v_hat = v_t / (1 - beta2^t)
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

Plain "Adam + L2 regularization" implements weight decay by adding `lambda * theta` directly to the gradient `g_t` before it ever reaches the momentum/variance estimates above — which means the decay term itself gets adaptively rescaled by `v_hat`, exactly like the gradient does. **AdamW** (Loshchilov & Hutter, 2019) decouples this: it applies weight decay as a separate, direct shrinkage of the parameter, *outside* the adaptive-learning-rate machinery entirely:

```
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * theta_{t-1}
```

The difference is subtle but measurable: with plain Adam+L2, parameters with a large historical gradient variance (large `v_hat`) get *less* effective weight decay than they "should," since the decay term is divided by the same `sqrt(v_hat)` as the gradient. AdamW's decoupled decay applies the *same* proportional shrinkage to every parameter regardless of its gradient history — which empirically generalizes better, and is why essentially every modern LLM pretraining run uses AdamW rather than plain Adam. `example.py` implements this exact update rule from scratch and checks it against `torch.optim.AdamW`.

## 4. Learning-rate schedules: warmup then decay

Real pretraining runs never use a constant learning rate. The standard recipe has two phases:

```
lr(step) = lr_max * (step / warmup_steps)                                              if step < warmup_steps
lr(step) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))                  otherwise
    where progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

- **Warmup** (linearly ramping the learning rate up from ~0 to `lr_max` over the first several hundred/thousand steps) matters because early in training, weights are randomly initialized and gradients are large and noisy — taking a full-sized optimizer step immediately can push the model into a bad region it never recovers from. A small, growing learning rate lets the model's early statistics (like Adam's own `m`/`v` running estimates, which start at zero and are themselves still "warming up") stabilize before the full step size is trusted.
- **Cosine decay** (or, more simply, linear decay) after warmup smoothly reduces the learning rate for the rest of training, converging toward a small (often near-zero) final value — large steps make fast early progress; small steps late in training let the model settle into a sharper, better-generalizing minimum instead of bouncing around a wide one.

`example.py` implements and tabulates exactly this warmup+cosine schedule.

## Video Script Outline

1. Motivation — "two implementation details that quietly make or break every real training run"
2. fp32 vs fp16 vs bf16: range vs. precision, and why bf16 won
3. The underflow problem, demonstrated with a concrete tiny gradient value
4. Loss scaling: scale before backward, unscale before the optimizer step
5. Adam recap, then AdamW's decoupled weight decay, side by side
6. Warmup + cosine decay, and the intuition for each phase
7. Walkthrough of `example.py` — AdamW from scratch vs. `torch.optim.AdamW`, the LR schedule table, and the fp16 underflow/loss-scaling demo
8. Recap + preview of Lesson 5: everything from Lessons 1-4 combined into one real (if tiny) training run

## Further Reading

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW)
- Micikevicius et al. (2018), *Mixed Precision Training* (the original fp16 + loss-scaling recipe)
- Kalamkar et al. (2019), *A Study of BFLOAT16 for Deep Learning Training*
- Loshchilov & Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine annealing schedules)

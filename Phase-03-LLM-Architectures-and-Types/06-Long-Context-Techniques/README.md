# Long-Context Techniques

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `06-Long-Context-Techniques`

## Why this matters

Two unresolved problems have been flagged and deferred since earlier in this course: self-attention's `O(T²)` cost ([Phase 01 §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity)), and absolute positional encodings' inability to generalize past the lengths seen during training ([Phase 02 Lesson 3 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md#5-learned-positional-embeddings)). This lesson resolves both, with the two techniques that essentially every modern long-context LLM actually uses: RoPE and sliding-window/local attention, plus ALiBi as an instructive alternative.

## What this lesson covers

- RoPE (Rotary Position Embedding): baking relative position directly into attention
- ALiBi: a parameter-free alternative that biases attention scores by distance
- Sliding-window (local) attention: trading full context for linear-time compute
- How these combine in real deployed models (previewed further in [Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md))

## 1. RoPE: rotating Q and K instead of adding to the input

Recall [Phase 02 Lesson 3 §4](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md#4-the-relative-position-trick): sinusoidal encodings have the property that `PE(pos+k)` is a fixed rotation of `PE(pos)`, which a linear layer *could in principle* learn to exploit for relative-position reasoning — but nothing forces it to. **RoPE** (Su et al., 2021) makes this the *only* option, by baking the rotation directly into the attention mechanism itself rather than adding a separate vector to the input:

```
q_rotated = Rotate(q, position_i)
k_rotated = Rotate(k, position_j)
q_rotated · k_rotated  depends ONLY on (position_i - position_j), never on the absolute positions
```

This uses exactly the same per-dimension-pair rotation matrices from [Phase 02 Lesson 3's `example.py`](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/example.py) — the only difference is *where* the rotation is applied: to `Q` and `K` themselves (inside every attention computation, at every layer) rather than once, additively, at the input embedding. Because relative position is now a mathematical guarantee of the dot product itself, RoPE-based models tend to generalize better to sequence lengths beyond what they were trained on than absolute-position schemes do. RoPE is used by LLaMA, Mistral, and most modern open LLMs.

## 2. ALiBi: no position embeddings at all

Press, Smith, Lewis (2021) took a different, strikingly simple approach: **don't encode position anywhere in the input or in Q/K at all.** Instead, directly subtract a distance-proportional penalty from the raw attention scores, before the softmax:

```
scores[i, j] = (q_i · k_j) - m · |i - j|      (only for j <= i, under a causal mask)
```

`m` is a fixed, head-specific slope (different heads get different, geometrically-spaced slopes, so some heads focus more locally and others more broadly). There are **zero learned parameters** for position — the penalty is a fixed function of distance, computed once and reused for any sequence length, including lengths far beyond training. This "train short, test long" robustness was ALiBi's specific selling point.

## 3. Sliding-window (local) attention: fixing the compute, not just position

RoPE and ALiBi both address *positional generalization* — neither touches the underlying `O(T²)` compute cost. **Sliding-window attention** attacks the compute problem directly: restrict each token to attending only to a fixed-size window of `w` nearby tokens (plus, in many implementations, a handful of designated "global" tokens that everyone can see), instead of the entire sequence:

```
Full attention:            each token attends to all T tokens      -> O(T^2)
Sliding-window attention:  each token attends to only w tokens      -> O(T * w), linear in T
```

Mistral popularized this at production scale, combined with RoPE, showing that most of a long sequence's useful context for any given token is genuinely local most of the time — and that stacking several sliding-window layers still lets information *eventually* propagate across the whole sequence (layer `l`'s window of size `w` combined with layer `l-1`'s already gives an effective receptive field of `~2w` after two layers, growing with depth), similar in spirit to how stacking small convolutional kernels builds a large effective receptive field in a CNN.

## 4. How these get combined in practice

Real long-context models mix and match: RoPE for relative-position-aware, length-generalizing attention, sometimes combined with sliding-window attention in some (or all) layers to control compute cost directly, occasionally alongside a few full-attention layers to preserve some genuinely global, unrestricted context. [Lesson 7's survey](../07-Survey-of-Popular-Open-LLMs/README.md) will show exactly which real, open models use which combination.

## Video Script Outline

1. Motivation — "two problems flagged since Phase 01/02, finally resolved"
2. RoPE: rotate Q/K, not add to input — relative position becomes mathematically guaranteed
3. ALiBi: skip position embeddings entirely, bias the scores directly by distance
4. Sliding-window attention: O(T²) -> O(T*w), and how stacked layers still see far
5. Walkthrough of `example.py` — implement RoPE and verify the relative-position property directly on real Q/K vectors; implement ALiBi's bias matrix; measure sliding-window attention's linear vs. full attention's quadratic growth
6. Recap + pointer to [Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md) for which real models use which combination

## Further Reading

- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE)
- Press, Smith, Lewis (2021), *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (ALiBi)
- Jiang et al. (2023), *Mistral 7B* (sliding-window attention at production scale)
- Beltagy, Peters, Cohan (2020), *Longformer: The Long-Document Transformer* (an earlier, influential local + global attention pattern)

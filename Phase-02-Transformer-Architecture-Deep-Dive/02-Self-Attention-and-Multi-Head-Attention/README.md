# Self-Attention and Multi-Head Attention

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `02-Self-Attention-and-Multi-Head-Attention`

## Why this matters

[Phase 01: Introduction to Transformers](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md) gave you a minimal, untrained self-attention layer to build intuition. This lesson makes it rigorous and complete: *why* the scores get divided by `√d_k`, how multiple attention "heads" work and why they help, and how to block a decoder from cheating by looking at future tokens. By the end, you'll have the exact, production-grade attention mechanism used inside every Transformer in this course, now built properly in PyTorch.

## What this lesson covers

- Scaled dot-product attention, and precisely why the scaling factor is needed
- Multi-head attention: splitting into subspaces, running attention in parallel, recombining
- Causal (look-ahead) masking for decoder-style models
- Padding masks for batched, variable-length sequences
- Building both in PyTorch as reusable `nn.Module`s

## 1. Scaled dot-product attention

Recap from [Phase 01 §1](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#1-the-key-insight-drop-the-recurrence-keep-the-attention) and the [Seq2Seq attention lesson](../../Phase-01-Language-Modeling-Foundations/04-Seq2Seq-and-Attention/README.md#5-this-is-already-queryknowledgevalue):

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

- `Q, K, V` — matrices of shape `(T, d_k)` (or `(T, d_v)` for `V`), one row per token
- `Q Kᵀ` — every query dotted against every key: an `(T, T)` matrix of raw similarity scores
- `softmax(·)` — turn each row into a probability distribution over "which tokens matter" ([Phase 00 §4](../../Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md#4-probability))
- `· V` — take the weighted average of value vectors according to that distribution

## 2. Why divide by `√d_k`?

This is the detail Phase 01's preview skipped. Assume `Q` and `K`'s entries are independent random values with mean 0 and variance 1. The dot product `q · k = Σᵢ qᵢkᵢ` sums `d_k` independent terms, so **its variance grows linearly with `d_k`** — for a large `d_k` (say 64 or 128), raw dot products can have quite large magnitude. Feed large-magnitude scores into `softmax`, and it saturates: one score dominates completely, the gradient of `softmax` becomes nearly zero almost everywhere else, and training stalls. Dividing by `√d_k` exactly cancels that variance growth (since `Var(q·k / √d_k) = Var(q·k) / d_k`), keeping the scores in a well-behaved range regardless of dimension. `example.py` demonstrates this saturation effect directly, numerically.

## 3. Multi-head attention

A single attention computation forces the model to blend *all* the relevant relationships between tokens into one weighting per pair. Multi-head attention instead **runs several smaller attention computations in parallel**, each in its own learned subspace, so different heads are free to specialize (one might learn to track "which noun does this adjective modify," another "the previous occurrence of this word," etc. — patterns real trained models are observed to learn):

```
head_i = Attention(Q Wᵢ^Q, K Wᵢ^K, V Wᵢ^V)          for i = 1..h
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

If `d_model = 512` and `h = 8` heads, each head works in `d_k = d_model / h = 64` dimensions — so multi-head attention has **the same total compute and parameter budget** as one big attention computation over the full `d_model`, just factored into `h` independent, narrower views.

## 4. Causal masking

A decoder generating text one token at a time must never let position `t` attend to positions `> t` — that would be looking at the answer before generating it. This is enforced by **masking**: before the softmax, set every "future" score to `-∞` (in practice, a very large negative number), so `softmax` assigns it a weight of essentially `0`:

```
scores[i, j] = -inf   whenever j > i
```

This turns full self-attention into **causal self-attention**, the exact mechanism every decoder-only model ([Phase 03: GPT family](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md)) uses, and what makes [Phase 02 Lesson 6's mini-GPT](../06-Mini-Transformer-From-Scratch/README.md) generate text autoregressively.

## 5. Padding masks

When batching sequences of different lengths together, shorter ones get padded with a `<pad>` token ([Lesson 1 §6](../01-Tokenization/README.md#6-special-tokens)). A **padding mask** similarly sets scores involving pad positions to `-∞`, so the model never wastes attention weight on meaningless filler tokens and gradients never flow through them.

## Video Script Outline

1. Motivation — "the untrained toy from last phase, made rigorous"
2. Scaled dot-product attention, formula on screen
3. Live demo: unscaled vs. scaled scores -> softmax saturation, side by side
4. Multi-head attention: split -> parallel attention -> concat -> project
5. Causal masking: draw the upper-triangular mask, tie to "can't see the future"
6. Walkthrough of `example.py` — PyTorch implementation, shapes, and the masking demo
7. Recap + pointer to Positional Encoding (attention still has no idea what order tokens came in!)

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.2 (this lesson is a direct expansion of it)
- Jay Alammar, *The Illustrated Transformer* — the multi-head attention diagrams especially
- PyTorch docs: `torch.nn.MultiheadAttention` (the library version of what we build by hand here)

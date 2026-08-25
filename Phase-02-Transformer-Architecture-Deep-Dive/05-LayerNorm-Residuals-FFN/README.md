# Layer Norm, Residuals and Feed-Forward Sublayers

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `05-LayerNorm-Residuals-FFN`

## Why this matters

The previous lesson used residual connections (`x = x + sublayer(x)`), `LayerNorm`, and a feed-forward block without pausing to justify any of them — they were treated as "the boring plumbing that makes the architecture work." This lesson stops and actually explains each one. None of them are optional details: without residual connections, real Transformers (12, 48, even 100+ layers deep) simply do not train; without LayerNorm, training is unstable; and the feed-forward sublayer, despite getting far less attention than self-attention, holds roughly two-thirds of a typical Transformer layer's parameters.

## What this lesson covers

- Residual (skip) connections and why they enable very deep networks
- Layer Normalization: the formula and why it's needed
- LayerNorm vs. BatchNorm — why Transformers specifically use LayerNorm
- Pre-LN vs. Post-LN architectures
- The position-wise feed-forward network, and why it holds so many parameters

## 1. Residual connections: why depth doesn't kill gradients

Recall the vanishing-gradient story from [Phase 01: RNNs, LSTMs and GRUs](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/README.md#3-vanishing-and-exploding-gradients): repeatedly composing transformations tends to shrink gradients as they backpropagate through many layers. A plain deep stack of Transformer sublayers, `x = sublayer(x)`, would have the exact same problem — 24 or 96 layers deep, gradients reaching the earliest layers could vanish entirely.

The fix (He et al., 2015, originally for CNNs, adopted directly by the Transformer): instead of *replacing* `x` with `sublayer(x)`, **add** the sublayer's output to the original input:

```
x = x + Sublayer(x)
```

During backpropagation, the gradient of a sum is the sum of the gradients — so the gradient flowing backward through this connection has a direct, unimpeded path (`+ x`'s gradient is just `1`) straight back to earlier layers, *in addition to* whatever path flows through `Sublayer`. Even if `Sublayer`'s own gradient contribution is small, the identity path guarantees signal still reaches early layers. This one addition is a major part of why Transformers can be stacked far deeper than pre-residual architectures ever could. `example.py` measures this effect directly, extending the same gradient-flow measurement technique from [Phase 01 Lesson 3](../../Phase-01-Language-Modeling-Foundations/03-RNN-LSTM-GRU/example.py).

## 2. Layer Normalization

Deep networks are notoriously sensitive to the *scale* of activations flowing through them — values that grow or shrink layer over layer destabilize training. **Layer Normalization** (Ba, Kiros, Hinton, 2016) re-centers and re-scales each individual token's activation vector to have mean 0 and variance 1 (then applies a learned scale `γ` and shift `β` so the network can undo the normalization if that's actually better for a given layer):

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

where `μ` and `σ²` are the mean and variance computed **across the features of a single token's vector** (not across the batch, and not across other tokens).

## 3. Why LayerNorm, not BatchNorm?

BatchNorm — ubiquitous in CNNs — normalizes across the **batch dimension**: for each feature, compute mean/variance over every example currently in the batch. Two problems make this a poor fit for sequence models:

- **Variable sequence lengths.** A batch of sentences of very different lengths (with padding) makes batch statistics noisy and dependent on how padding is handled.
- **Sequential/autoregressive generation.** At inference time you may process one token (or one very small batch) at a time — batch statistics become meaningless or unavailable entirely.

LayerNorm sidesteps both: its statistics are computed **per token**, independent of batch size or other tokens' presence, making it equally well-defined whether you're processing a batch of 512 sequences during training or generating one token at a time during inference.

## 4. Pre-LN vs. Post-LN

The original Transformer paper (and [Lesson 4](../04-Transformer-Encoder-Decoder/README.md)'s presentation) applies `LayerNorm` **after** adding the residual — "Post-LN": `x = LayerNorm(x + Sublayer(x))`. In practice, as models got deeper, this was found to make training unstable without a carefully tuned learning-rate warmup. Most modern LLMs (GPT-2 onward) instead use **Pre-LN**: normalize *before* the sublayer, and add the residual around the whole thing:

```
Pre-LN:  x = x + Sublayer(LayerNorm(x))
```

Pre-LN keeps the residual path a truly clean, unmodified identity connection from input to output (LayerNorm only ever touches the branch going *into* the sublayer), which empirically gives much more stable gradients in very deep stacks, at a small cost in final model quality relative to a well-tuned Post-LN model. This is a small-looking architectural choice with an outsized effect on trainability at scale.

## 5. The position-wise feed-forward network (FFN)

Attention is the only sublayer where tokens exchange information with each other. The FFN sublayer does the opposite: it processes **each token's vector independently and identically** — a small 2-layer MLP applied at every position with the exact same weights:

```
FFN(x) = W₂ · activation(W₁ x + b₁) + b₂
```

The original paper used ReLU; GPT-2 onward typically uses **GELU** (recall [Phase 00: Neural Networks Basics §2](../../Phase-00-Prerequisites/02-Neural-Networks-Basics/README.md#2-activation-functions)). The hidden dimension `d_ff` is conventionally **4x** `d_model` (e.g. `d_model=768` -> `d_ff=3072` in GPT-2-small) — this expand-then-contract shape gives the sublayer substantial capacity to transform each token's representation. Because `W₁` and `W₂` are full `(d_model, d_ff)` matrices, the FFN sublayer typically holds **roughly two-thirds of a Transformer layer's total parameters** — attention gets most of the conceptual attention (pun intended), but the FFN is doing more of the raw parameter-counted "work."

## Video Script Outline

1. Motivation — "the boring plumbing that makes 96-layer models actually trainable"
2. Residual connections: the gradient-highway intuition, tie back to Phase 01's vanishing gradients
3. LayerNorm formula, and why per-token (not per-batch) normalization fits sequences
4. Pre-LN vs. Post-LN, and why modern models moved to Pre-LN
5. The FFN sublayer: expand-then-contract, and the parameter-count surprise
6. Walkthrough of `example.py` — measure the residual-connection gradient-flow effect directly, verify a hand-rolled LayerNorm against PyTorch's, and count FFN vs. attention parameters
7. Recap: every piece is now built -> next lesson assembles a full trainable mini-GPT

## Further Reading

- He, Zhang, Ren, Sun (2015), *Deep Residual Learning for Image Recognition* (ResNets — origin of the residual-connection idea)
- Ba, Kiros, Hinton (2016), *Layer Normalization*
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* (the Pre-LN vs. Post-LN training-stability analysis)
- Vaswani et al. (2017), *Attention Is All You Need*, Section 3.3 (the FFN sublayer)

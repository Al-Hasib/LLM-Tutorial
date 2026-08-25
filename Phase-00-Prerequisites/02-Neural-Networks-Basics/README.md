# Neural Networks Basics

**Phase:** [Prerequisites](../README.md) · **Topic folder:** `02-Neural-Networks-Basics`

## Why this matters

A Transformer is "just" a very deep, very specific arrangement of the exact same building block you're about to learn here: weighted sums, non-linear activations, and gradient descent. If you understand how a 2-layer network learns XOR by hand, you already understand — mechanically — how a 100-billion-parameter model learns to predict the next token. Scale changes; the mechanism does not.

## What this lesson covers

- The perceptron: the atomic unit of a neural network
- Activation functions and why non-linearity is non-negotiable
- The multi-layer perceptron (MLP) forward pass
- Loss functions: MSE and cross-entropy
- Backpropagation: the chain rule, applied layer by layer
- Gradient descent and its variants

## 1. The perceptron

A perceptron computes a weighted sum of its inputs, adds a bias, and applies an activation function:

```
z = w · x + b
a = φ(z)
```

- `x` — input vector
- `w` — learned weight vector
- `b` — learned bias (scalar)
- `φ` — activation function
- `a` — the neuron's output ("activation")

Stack many perceptrons side by side and you get a **layer**; stack layers and you get a **multi-layer perceptron (MLP)**.

## 2. Activation functions

Without a non-linear `φ`, stacking layers is pointless — a composition of linear functions is still just a linear function, so a 1000-layer network with no activations is exactly as expressive as a single layer. Common choices:

| Function | Formula | Notes |
|---|---|---|
| Sigmoid | `σ(z) = 1 / (1 + e⁻ᶻ)` | Squashes to `(0, 1)`; historically popular, saturates and kills gradients for large \|z\| |
| Tanh | `tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)` | Squashes to `(-1, 1)`; zero-centered, still saturates |
| ReLU | `max(0, z)` | Cheap, doesn't saturate for `z > 0`; the default for most deep nets |
| GELU | `z · Φ(z)` (Φ = Gaussian CDF) | Smooth ReLU variant; used inside every Transformer feed-forward block you'll build in Phase 02 |

## 3. Forward pass through an MLP

For a network with layers `1..L`:

```
a⁽⁰⁾ = x
z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = φ(z⁽ˡ⁾)
ŷ    = a⁽ᴸ⁾
```

Each layer's weight matrix `W⁽ˡ⁾` has shape `(units_in_layer_l, units_in_layer_l-1)` — exactly the matrix-multiplication shape rules from [Python and Math Refresher](../01-Python-and-Math-Refresher/README.md).

## 4. Loss functions

- **Mean Squared Error (regression)**: `L = (1/n) Σ (ŷᵢ − yᵢ)²`
- **Cross-entropy (classification)**: `L = −Σ yᵢ log(ŷᵢ)` — this is exactly the next-token loss every LLM uses, just over a vocabulary-sized output instead of a handful of classes.

## 5. Backpropagation

Backprop is the chain rule ([Python and Math Refresher, §3](../01-Python-and-Math-Refresher/README.md#3-calculus)) applied systematically from the loss backward to every weight:

1. Compute the forward pass, caching every intermediate `z⁽ˡ⁾` and `a⁽ˡ⁾`.
2. Compute `∂L/∂a⁽ᴸ⁾` at the output.
3. For each layer `l` from `L` down to `1`:
   - `∂L/∂z⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ ⊙ φ'(z⁽ˡ⁾)` (⊙ = elementwise product)
   - `∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · a⁽ˡ⁻¹⁾ᵗ`
   - `∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾`
   - `∂L/∂a⁽ˡ⁻¹⁾ = W⁽ˡ⁾ᵗ · ∂L/∂z⁽ˡ⁾` (propagate the error one layer further back)

Every deep learning framework (PyTorch included — see [PyTorch Fundamentals](../04-PyTorch-Fundamentals/README.md)) automates exactly this recursion via **autograd**, so you never hand-derive it again after this lesson — but you should watch it happen in raw code at least once.

## 6. Gradient descent

Once we have `∂L/∂θ` for every parameter `θ`, we nudge it downhill:

```
θ ← θ − η · ∂L/∂θ
```

`η` (eta) is the **learning rate**. Variants you'll meet again later in the course:

- **Batch GD**: use the whole dataset per update (slow, stable)
- **Stochastic GD (SGD)**: one example per update (fast, noisy)
- **Mini-batch SGD**: a small batch per update — what's actually used in practice, LLMs included
- **Momentum / Adam**: accumulate a running average of past gradients to smooth and accelerate convergence (Adam/AdamW is the default optimizer for training LLMs — covered in [Phase 04: Mixed Precision and Optimization](../../Phase-04-Pretraining-LLMs/04-Mixed-Precision-and-Optimization/README.md))

## Video Script Outline

1. Motivation — "an LLM layer is just this, repeated and scaled"
2. Perceptron → activation functions, with a plot of each activation shape
3. Stack into an MLP, walk through the forward-pass equations
4. Derive backprop for a tiny 2-layer network on the whiteboard
5. Walkthrough of `example.py` — train a from-scratch NumPy MLP on XOR, watch the loss curve fall
6. Recap + pointer to Introduction to NLP next

## Further Reading

- Michael Nielsen, *Neural Networks and Deep Learning* (free online book)
- 3Blue1Brown — *Neural Networks* video series (esp. the backpropagation episodes)
- Andrej Karpathy — *The spelled-out intro to neural networks and backpropagation: building micrograd*
- Goodfellow, Bengio, Courville — *Deep Learning*, Ch. 6

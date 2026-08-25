# Python and Math Refresher

**Phase:** [Prerequisites](../README.md) · **Topic folder:** `01-Python-and-Math-Refresher`

## Why this matters

Every later lesson in this course boils down to vectors, matrices, derivatives, and probabilities being pushed through code. You don't need a math degree to understand LLMs, but you do need enough linear algebra, calculus, and probability *intuition* to read an equation from a paper and immediately picture what shape of data is flowing where. This lesson builds exactly that toolbox, plus the handful of Python idioms you'll see reused in every `example.py` in this repo.

## What this lesson covers

- Python idioms used throughout the course (comprehensions, unpacking, classes)
- Linear algebra: vectors, matrices, dot products, matrix multiplication, norms
- Calculus: derivatives, partial derivatives, gradients, the chain rule
- Probability: distributions, expectation, softmax, log-likelihood
- A shared notation cheat sheet for the rest of the series

## 1. Python idioms you'll keep seeing

```python
# List/dict comprehensions instead of manual loops
squares = [x ** 2 for x in range(5)]
vocab_to_id = {word: i for i, word in enumerate(["the", "cat", "sat"])}

# enumerate + zip for parallel iteration
for i, (word, idx) in enumerate(zip(["a", "b"], [0, 1])):
    ...

# Unpacking
batch_size, seq_len, hidden_dim = tensor.shape

# A minimal class, the shape every nn.Module will follow later
class Layer:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError
```

The `__call__` → `forward` pattern above is exactly how `torch.nn.Module` works — we're pre-building the mental model for [PyTorch Fundamentals](../04-PyTorch-Fundamentals/README.md).

## 2. Linear algebra

LLMs are, at their core, an enormous number of matrix multiplications with some non-linearities sprinkled in. You need to be fluent in this small toolkit:

- **Scalar**: a single number.
- **Vector** `v ∈ ℝⁿ`: an ordered list of numbers (e.g. a word embedding).
- **Matrix** `A ∈ ℝᵈˣⁿ`: rows × columns (e.g. a weight matrix mapping an `n`-dim input to a `d`-dim output).
- **Tensor**: the generalization to 3+ dimensions (e.g. `[batch, sequence, hidden]`).

**Dot product** of two vectors of the same length:

```
a · b = Σᵢ aᵢ bᵢ
```

This single operation is the heart of attention: "how similar are these two vectors" is measured with a dot product.

**Matrix multiplication**: `A (d×n) @ B (n×m) = C (d×m)`. The inner dimensions must match — this is the single most common bug source in every neural network, so get comfortable checking shapes before checking values.

**Transpose** `Aᵗ` flips rows and columns — used constantly to make shapes line up (`Q @ Kᵗ` in attention, for instance).

**Norm** measures a vector's "size":
- L2 (Euclidean): `‖v‖₂ = √(Σ vᵢ²)`
- L1: `‖v‖₁ = Σ |vᵢ|`

## 3. Calculus

Training a neural network is just: compute a loss, then figure out which direction to nudge every parameter to make that loss smaller. That "which direction" question is answered entirely by derivatives.

- **Derivative** `f'(x)`: instantaneous rate of change of `f` at `x`.
- **Partial derivative** `∂f/∂x`: rate of change with respect to *one* variable, holding the others fixed — needed because a model has millions/billions of parameters.
- **Gradient** `∇f`: the vector of all partial derivatives — it points in the direction of steepest increase, so training moves *against* it (gradient **descent**).
- **Chain rule**: if `y = f(g(x))`, then `dy/dx = f'(g(x)) · g'(x)`. Backpropagation (next lesson) is nothing more than the chain rule applied automatically, layer by layer, across the whole network.

## 4. Probability

- A **probability distribution** over a set of outcomes assigns each outcome a non-negative weight that sums to 1.
- **Expectation** `E[X] = Σ x · P(x)`: the average outcome, weighted by likelihood.
- **Softmax** turns arbitrary real-valued scores ("logits") into a probability distribution:

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Every LLM ends its forward pass with a softmax over the vocabulary — "logits" become "the probability of each possible next token."

- **Log-likelihood / cross-entropy**: instead of maximizing the probability the model assigns to the correct token, we (equivalently, and more numerically stably) minimize the negative log of that probability. This is the loss function used in virtually every LLM you will train in this course.

## Notation cheat sheet (used across this repo)

| Symbol | Meaning |
|---|---|
| `x`, `v` | scalar / vector |
| `W`, `A` | matrix |
| `Wᵗ` | transpose of `W` |
| `d_model` | hidden/embedding dimension |
| `n`, `T` | sequence length |
| `∇` | gradient |
| `θ` | model parameters (weights) |
| `L` | loss |

## Video Script Outline

1. Motivation — "an LLM is just linear algebra + calculus + probability at scale," preview why each piece matters
2. Linear algebra walkthrough with shape diagrams (vectors → matrices → dot products)
3. Calculus: derivative → gradient → chain rule, tie directly to "how training works"
4. Probability: softmax and log-likelihood, tie directly to "how the model outputs a next-token guess"
5. Walkthrough of `example.py` — run every concept as real code
6. Recap + notation cheat sheet on screen + pointer to Neural Networks Basics next

## Further Reading

- 3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus* video series
- Gilbert Strang, *Introduction to Linear Algebra*
- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (free PDF)
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Ch. 1–2 (probability refresher)

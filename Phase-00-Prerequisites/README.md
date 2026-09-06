# Prerequisites

[← Back to curriculum index](../README.md)

Brush up on the math, Python, and deep-learning basics needed before touching a Transformer.

## The path through this phase

Four lessons, and only one of them is about text. The goal is that nothing in Phase 02 — where a Transformer gets taken apart — is unfamiliar *mathematics* or unfamiliar *PyTorch*, so that all your attention can go on the architecture itself.

```mermaid
flowchart LR
    A["01 · Python & math refresher<br/>vectors · matrices · gradients<br/>the chain rule"] --> B["02 · neural network basics<br/>forward pass · loss · backprop"]
    B --> C["04 · PyTorch fundamentals<br/>tensors · autograd · nn.Module"]
    A --> C
    D["03 · introduction to NLP<br/>text as data: tokens,<br/>bag-of-words, TF-IDF"] --> NEXT["Phase 01 ·<br/>Language Modeling Foundations"]
    C --> NEXT
```

Skip a lesson if you already know it — they are independent except that 02 assumes 01, and 04 is easier after both.

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [Python and Math Refresher](01-Python-and-Math-Refresher/README.md) |
| 02 | [Neural Networks Basics](02-Neural-Networks-Basics/README.md) |
| 03 | [Introduction to NLP](03-Intro-to-NLP/README.md) |
| 04 | [PyTorch Fundamentals](04-PyTorch-Fundamentals/README.md) |

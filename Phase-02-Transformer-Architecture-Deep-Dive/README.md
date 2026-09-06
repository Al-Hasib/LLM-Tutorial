# Transformer Architecture Deep Dive

[← Back to curriculum index](../README.md)

Take the Transformer apart piece by piece and rebuild a mini version from scratch in PyTorch.

## The path through this phase

Every component of a Transformer, in the order you need it to build one — ending with a working mini-GPT written from scratch, then the efficiency work that makes real context lengths possible.

```mermaid
flowchart LR
    T["01 · tokenization<br/>text → token ids"] --> A["02 · self-attention<br/>and multi-head attention"]
    A --> P["03 · positional encoding<br/>attention alone has<br/>no sense of order"]
    P --> ED["04 · encoder–decoder<br/>how the blocks assemble"]
    ED --> N["05 · LayerNorm · residuals · FFN<br/>what makes a deep stack trainable"]
    N --> M["06 · build a mini-GPT<br/>from scratch, in PyTorch"]
    M --> E["07 · efficient attention<br/>FlashAttention · sparse · linear"]
```

Lessons 01–05 are the parts; Lesson 06 is where they become a model you can train and sample from. Lesson 07 then asks what has to change when the sequence gets long.

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [Tokenization](01-Tokenization/README.md) |
| 02 | [Self-Attention and Multi-Head Attention](02-Self-Attention-and-Multi-Head-Attention/README.md) |
| 03 | [Positional Encoding](03-Positional-Encoding/README.md) |
| 04 | [Transformer Encoder-Decoder Architecture](04-Transformer-Encoder-Decoder/README.md) |
| 05 | [Layer Norm, Residuals and Feed-Forward Sublayers](05-LayerNorm-Residuals-FFN/README.md) |
| 06 | [Building a Mini-Transformer / Mini-GPT From Scratch](06-Mini-Transformer-From-Scratch/README.md) |
| 07 | [Efficient Attention: FlashAttention, Sparse and Linear Attention](07-Efficient-Attention-FlashAttention-and-Approximations/README.md) |

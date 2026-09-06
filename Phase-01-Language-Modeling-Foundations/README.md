# Language Modeling Foundations

[← Back to curriculum index](../README.md)

Understand what a language model actually is and how the field evolved from n-grams to attention before Transformers existed.

## The path through this phase

This phase is deliberately historical: each lesson introduces a solution, and the next lesson shows the problem that solution still had. By the end, the Transformer arrives not as a novelty but as the obvious response to a specific bottleneck you have already seen fail.

```mermaid
flowchart LR
    A["01 · what a language model IS<br/>P(next token | context)"] --> B["02 · word embeddings<br/>meaning as geometry"]
    B --> C["03 · RNNs · LSTMs · GRUs<br/>memory across a sequence —<br/>and why it fades"]
    C --> D["04 · seq2seq + attention<br/>the one-vector bottleneck,<br/>and the fix that removed it"]
    D --> E["05 · the Transformer<br/>keep attention · drop recurrence<br/>· parallelize everything"]
```

## Topics in this phase

| # | Topic |
|---|-------|
| 01 | [What is a Language Model](01-What-is-a-Language-Model/README.md) |
| 02 | [Word Embeddings](02-Word-Embeddings/README.md) |
| 03 | [RNNs, LSTMs and GRUs](03-RNN-LSTM-GRU/README.md) |
| 04 | [Sequence-to-Sequence and Attention](04-Seq2Seq-and-Attention/README.md) |
| 05 | [Introduction to Transformers](05-Intro-to-Transformers/README.md) |

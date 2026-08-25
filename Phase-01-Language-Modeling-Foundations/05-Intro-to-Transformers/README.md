# Introduction to Transformers

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `05-Intro-to-Transformers`

## Why this matters

This lesson is the hinge point of the entire course. Everything up to now — n-grams, embeddings, RNNs, attention bolted onto an RNN — was the field groping toward an idea that, once stated plainly, seems almost obvious in hindsight: **if attention is powerful enough to replace an RNN's memory, why keep the RNN at all?** Vaswani et al.'s 2017 paper, bluntly titled *"Attention Is All You Need,"* answered exactly that. This lesson gives you the big picture before [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) takes the architecture apart piece by piece and rebuilds it from scratch.

## What this lesson covers

- Why removing recurrence entirely was the key insight
- Self-attention: attention applied *within* a single sequence
- Parallelization: the practical reason Transformers won
- Long-range dependencies without the vanishing-gradient path length problem
- A first look at the overall architecture shape
- The complexity trade-off Transformers introduced
- A roadmap for the deep dive in Phase 02

## 1. The key insight: drop the recurrence, keep the attention

[Sequence-to-Sequence and Attention](../04-Seq2Seq-and-Attention/README.md) used attention as a *supplement* to an RNN — the decoder was still a recurrent network, just one that got to peek back at all encoder states instead of relying on a single compressed vector. Vaswani et al. asked: what if attention is doing all the real work already? Their answer was to remove the RNN entirely and build a model out of **self-attention** — attention where the queries, keys, and values *all come from the same sequence.*

```
Seq2Seq + attention:  decoder (recurrent) queries -> encoder (recurrent) states
Self-attention:       every position in a sequence queries -> every position in that SAME sequence
```

Every token gets to directly look at every other token in the sequence — no more information having to survive being passed hand-to-hand through dozens of recurrent timesteps.

## 2. Why this was the practical breakthrough: parallelization

Recall from [RNNs, LSTMs and GRUs §2](../03-RNN-LSTM-GRU/README.md#2-backpropagation-through-time-bptt): an RNN *must* process timestep `t-1` before it can process timestep `t`. That sequential chain is a hard constraint that no amount of clever engineering removes.

Self-attention has no such constraint. Computing how every token attends to every other token is, as you'll see in `example.py`, a small number of large matrix multiplications over the *entire sequence at once* — exactly the kind of computation GPUs are built to do extremely fast in parallel. This is arguably **the** reason Transformers won: not that they're smarter per se, but that they can be trained on vastly more data in the same wall-clock time, and scale (see [Phase 03: Scaling Laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md)) turned out to matter enormously.

## 3. Long-range dependencies, structurally

In an RNN, the "path length" between two tokens `i` and `j` in the computation graph is `|i - j|` steps — which is exactly why gradients (and information) decayed over distance in [Lesson 3](../03-RNN-LSTM-GRU/README.md#3-vanishing-and-exploding-gradients). In self-attention, the path length between *any* two tokens is **always 1** — every token computes a direct dot product against every other token, regardless of how far apart they are in the sequence. This structurally removes the vanishing-signal-over-distance problem, rather than just mitigating it the way LSTM gating did.

## 4. The shape of the architecture (preview)

The original Transformer is an encoder-decoder model, closely mirroring the Seq2Seq shape from Lesson 4, but with every recurrent layer replaced by self-attention + a small feedforward network:

```
Input tokens -> [Embedding + Positional Encoding]
             -> [ N x (Self-Attention -> Feed-Forward) ] encoder stack
             -> [ N x (Self-Attention -> Cross-Attention -> Feed-Forward) ] decoder stack
             -> Output probabilities (softmax over vocabulary)
```

Two pieces here have no RNN analog and get their own full lessons next phase:

- **Positional Encoding**: self-attention has no inherent sense of word order (unlike an RNN, which processes tokens strictly in sequence) — order has to be injected explicitly. See [Phase 02: Positional Encoding](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md).
- **Multi-head attention**: instead of computing one attention pattern, compute several in parallel, each potentially learning to focus on different kinds of relationships. See [Phase 02: Self-Attention and Multi-Head Attention](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md).

## 5. The trade-off: quadratic complexity

Nothing is free. Because every token attends to every other token, self-attention costs `O(T²)` in both compute and memory for a sequence of length `T` — compared to an RNN's `O(T)`. For a while this was a fine trade (parallelism more than made up for it), but it's exactly why "how do we handle very long sequences cheaply" became its own major research area — see [Phase 03: Long-Context Techniques](../../Phase-03-LLM-Architectures-and-Types/06-Long-Context-Techniques/README.md) and [Phase 11: State Space Models](../../Phase-11-Advanced-and-Frontier-Topics/03-State-Space-Models-Mamba/README.md) later in this course.

## 6. Roadmap: what Phase 02 covers in depth

This lesson is intentionally the "big picture" version. Phase 02 will rebuild every piece from scratch: [Tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md), the exact [self-attention](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) math (including *why* it's scaled and *why* multiple heads help), [positional encoding](../../Phase-02-Transformer-Architecture-Deep-Dive/03-Positional-Encoding/README.md), the [full encoder-decoder architecture](../../Phase-02-Transformer-Architecture-Deep-Dive/04-Transformer-Encoder-Decoder/README.md), and finally [assembling a working mini-GPT from scratch](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md).

## Video Script Outline

1. Motivation — "one sentence rewrote the field: attention is all you need"
2. Self-attention vs. Seq2Seq-attention: same mechanism, no more RNN wrapper
3. Parallelization: draw the RNN dependency chain vs. the self-attention all-pairs matrix
4. Path length: O(1) vs O(distance) for information to travel between two tokens
5. High-level architecture diagram, naming the two genuinely new pieces (positional encoding, multi-head)
6. The O(T²) trade-off, briefly
7. Walkthrough of `example.py` — a minimal single-head self-attention layer, plus a toy timing comparison of sequential vs. parallel processing
8. Recap + the Phase 02 roadmap

## Further Reading

- Vaswani et al. (2017), *Attention Is All You Need*
- Jay Alammar, *The Illustrated Transformer* (jalammar.github.io) — the most-cited visual explainer of this architecture
- Jay Alammar, *The Illustrated Self-Attention* section within the above
- Sasha Rush et al., *The Annotated Transformer* — the paper re-implemented line by line in PyTorch

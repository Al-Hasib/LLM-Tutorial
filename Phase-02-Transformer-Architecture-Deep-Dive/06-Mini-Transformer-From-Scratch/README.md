# Building a Mini-Transformer / Mini-GPT From Scratch

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `06-Mini-Transformer-From-Scratch`

## Why this matters

This is the payoff for the entire phase. Every piece — [BPE tokenization](../01-Tokenization/README.md), [multi-head causal self-attention](../02-Self-Attention-and-Multi-Head-Attention/README.md), [positional encoding](../03-Positional-Encoding/README.md), the [decoder architecture](../04-Transformer-Encoder-Decoder/README.md), and [residuals/LayerNorm/FFN](../05-LayerNorm-Residuals-FFN/README.md) — gets assembled into one small, **fully working, trainable language model**: a decoder-only Transformer, exactly the architecture family every GPT-style LLM in this course belongs to ([Phase 03: Decoder-Only Models](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md)). You will train it from randomly initialized weights on real text and watch it go from producing gibberish to producing recognizable structure — the exact same training procedure (next-token prediction, cross-entropy loss, gradient descent) used to train every real LLM, just at a scale that fits in this lesson.

## What this lesson covers

- Why a decoder-only model drops cross-attention entirely
- Assembling token embedding + positional encoding + N decoder blocks + output head
- The training objective: next-token prediction, exactly as in [Phase 01: What is a Language Model](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md), now with a neural network instead of n-gram counts
- Autoregressive text generation
- What's genuinely different about a real, production-scale LLM (and what isn't)

## 1. From encoder-decoder to decoder-only

[Lesson 4](../04-Transformer-Encoder-Decoder/README.md#5-where-architectures-diverge-from-here) previewed this: a decoder-only model is the decoder stack from Lesson 4, with the cross-attention sublayer simply **removed** (there's no separate source sequence to attend to — every position attends causally to earlier positions in the *same* sequence):

```
x = x + CausalSelfAttention(LayerNorm(x))     # sublayer 1 (Pre-LN, per Lesson 5 §4)
x = x + FeedForward(LayerNorm(x))              # sublayer 2
```

Stack this block `N` times, and the "target sequence" and "source sequence" become the same thing: the model just predicts each next token from everything before it in one continuous stream. This is the entire architecture of GPT-1/2/3-family models.

## 2. The full model

```
token ids -> [token embedding + positional encoding]
          -> [ N x DecoderBlock(causal self-attention, FFN) ]
          -> LayerNorm
          -> Linear projection to vocabulary size
          -> softmax -> probability distribution over the next token
```

This is a direct, line-for-line combination of every prior lesson in this phase, minus the cross-attention sublayer.

## 3. Training objective: next-token prediction

Exactly the objective from [Phase 01 Lesson 1](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md#1-what-a-language-model-actually-is), just with a neural network estimating the conditional probabilities instead of raw n-gram counts:

```
Loss = CrossEntropy( model(tokens[:-1]), tokens[1:] )
```

Feed the model tokens `0..T-2`; its causal-masked output at each position `t` should predict token `t+1`. Because of the causal mask, this can be computed for *every* position in one parallel forward pass (no separate forward pass per position needed) — training a decoder-only Transformer on a sequence of length `T` gets `T` next-token-prediction training signals essentially for free from one pass.

## 4. Autoregressive generation

At inference time, there's no ground-truth "next token" to feed in — so generation happens one token at a time: run the model on the tokens so far, take the probability distribution at the *last* position, sample (or take the most likely) next token, append it, and repeat. This is identical in structure to the [bigram model's generation loop from Phase 01](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/example.py) — only the probability distribution being sampled from is now produced by a full Transformer instead of a count table.

> **Runtime note:** `example.py` trains for 2000 steps, which takes roughly 1-2 minutes on a CPU. Watching the loss fall from ~3.6 (near-random over a 29-character vocabulary) to ~0.1, and the generated text go from gibberish to fluent, grammatical sentences matching the training corpus, is the entire point — let it run.

## 5. What's different about a real LLM (and what isn't)

`example.py` trains a genuinely tiny model (a few hundred thousand parameters, a handful of layers, a context window of a few dozen characters) on a small chunk of text for a couple of minutes on a CPU. A production LLM differs in:

- **Scale**: billions of parameters, trillions of training tokens, thousands of GPUs — see [Phase 03: Scaling Laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) and [Phase 04: Pretraining LLMs](../../Phase-04-Pretraining-LLMs/README.md).
- **Tokenization**: a real byte-level BPE vocabulary of tens of thousands of tokens ([Lesson 1](../01-Tokenization/README.md)), not the character-level tokenizer used here for simplicity.
- **Training infrastructure**: distributed training, mixed precision, learning-rate schedules — [Phase 04](../../Phase-04-Pretraining-LLMs/README.md) in full.
- **What comes after pretraining**: instruction tuning and alignment ([Phase 05](../../Phase-05-Finetuning-LLMs/README.md), [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md)) before it behaves like an assistant.

What is **not** different: the core architecture and training loop in `example.py` — embedding, causal self-attention, feed-forward, residuals, LayerNorm, cross-entropy on next-token prediction, gradient descent — is *exactly* the same recipe, at every scale, all the way up.

## Video Script Outline

1. Motivation — "every piece from this entire phase, assembled into something that actually learns"
2. Decoder-only shape: cross-attention removed, everything else the same
3. Training objective recap: next-token prediction, now with a real neural network
4. Walkthrough of `example.py` — build the model, train it, watch the loss curve, generate text before vs. after training
5. Honest scale comparison: what's different about a real LLM, and what genuinely isn't
6. Recap of the whole phase + preview of Phase 03: the model architecture families built on this exact foundation

## Further Reading

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1 — the first decoder-only Transformer LM)
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (video) and the `nanoGPT` repository — the direct inspiration for this lesson's structure
- Sasha Rush et al., *The Annotated Transformer*

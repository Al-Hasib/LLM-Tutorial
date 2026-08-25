# RNNs, LSTMs and GRUs

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `03-RNN-LSTM-GRU`

## Why this matters

N-grams (Lesson 1) fail because their context window is fixed and tiny. Word embeddings (Lesson 2) fix *what a word means* but say nothing about *order or sequence*. Recurrent Neural Networks were the field's first serious attempt at a network that carries a memory forward through an arbitrarily long sequence — and for roughly a decade (2014-2017ish) they were the backbone of every serious NLP system, including Google Translate. Understanding exactly *why* they eventually lost to Transformers (their sequential, step-by-step nature) is what makes the payoff of self-attention in [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) click.

## What this lesson covers

- The vanilla RNN: recurrence, unrolling through time
- Backpropagation Through Time (BPTT) and why it's expensive/sequential
- The vanishing/exploding gradient problem
- LSTMs: the cell state and its three gates
- GRUs: a leaner alternative to the LSTM
- Why recurrence itself — not just the gradient problem — is what Transformers ultimately remove

## 1. The vanilla RNN

An RNN processes a sequence one element at a time, maintaining a **hidden state** `h_t` that's meant to summarize everything seen so far:

```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
y_t = W_hy h_t + b_y
```

The *same* weight matrices (`W_xh`, `W_hh`, `W_hy`) are reused at every timestep — this weight sharing is what lets an RNN handle sequences of any length with a fixed number of parameters. "Unrolling" the recurrence across `T` timesteps turns it into a very deep feedforward network (depth = sequence length) with tied weights at every layer.

## 2. Backpropagation Through Time (BPTT)

Training an RNN means backpropagating the loss through every one of those unrolled timesteps, applying the chain rule ([Phase 00 §3](../../Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md#3-calculus)) repeatedly through the recurrence. Two consequences fall directly out of this:

1. **It's inherently sequential.** You cannot compute `h_t` without first having `h_{t-1}` — there is no way to parallelize across timesteps during training or inference. This single fact is the main reason Transformers eventually replaced RNNs: modern GPUs are built for parallel work, and a sequential dependency chain wastes almost all of that capability.
2. **Gradients get multiplied by the same Jacobian over and over.** The gradient flowing back to an early timestep has passed through a `tanh'(·) · W_hh` factor once per intervening timestep.

## 3. Vanishing and exploding gradients

Because the same factor is multiplied at every timestep, gradients behave like a geometric series:

- If that factor's magnitude is consistently `< 1` (very common — `tanh'` is at most `1` and often much smaller), gradients shrink **exponentially** with sequence length: the **vanishing gradient problem**. Practically, this means a vanilla RNN essentially cannot learn dependencies more than ~10-20 steps apart — a word at the start of a paragraph has no real influence on training signal reaching the end.
- If that factor's magnitude is consistently `> 1`, gradients grow **exponentially**: the **exploding gradient problem**, usually fixed in practice with gradient clipping.

`example.py` measures this directly: it computes how much the final hidden state actually changes in response to a nudge to the very first input, for increasing sequence lengths.

## 4. LSTMs: gates and a protected cell state

The Long Short-Term Memory cell (Hochreiter & Schmidhuber, 1997) fixes vanishing gradients by adding a second recurrent quantity, the **cell state** `c_t`, that information can flow through with much less multiplicative decay, controlled by three learned **gates** (each a sigmoid, so it outputs values in `[0, 1]` acting as a soft on/off switch):

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # forget gate: how much of the old cell state to keep
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # input gate: how much new information to write in
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # candidate new content
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         # cell state update — mostly additive, not multiplicative!
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # output gate: how much of the cell state to expose
h_t = o_t ⊙ tanh(c_t)
```

The key insight: `c_t`'s update is **additive** (`f_t ⊙ c_{t-1} + ...`), not a repeated matrix multiplication like the vanilla RNN's hidden state. When the forget gate `f_t` is close to `1`, gradients can flow backward through many timesteps largely unchanged — this is exactly what `example.py` demonstrates numerically.

## 5. GRUs: a simpler alternative

The Gated Recurrent Unit (Cho et al., 2014) merges the forget and input gates into a single **update gate**, and drops the separate cell state entirely, folding everything into the hidden state:

```
z_t = σ(W_z · [h_{t-1}, x_t])                          # update gate
r_t = σ(W_r · [h_{t-1}, x_t])                          # reset gate
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])                 # candidate hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t                   # blend old and new
```

Fewer parameters than an LSTM, often comparable performance, and was a popular default in the mid-2010s. The core trick — a gated, largely-additive update instead of a pure matrix-multiply recurrence — is identical in spirit to the LSTM's.

## 6. What Transformers actually remove

It's tempting to think "LSTMs solved the RNN problem," but they only solved the *gradient* problem — the **sequential bottleneck** (you must process token `t-1` before token `t`) remains in LSTMs and GRUs too. That bottleneck is what made training on internet-scale data prohibitively slow for recurrent models, and it's the specific thing self-attention removes: every position can be processed **in parallel**, looking at every other position directly, no recurrence required. That's next: [Sequence-to-Sequence and Attention](../04-Seq2Seq-and-Attention/README.md).

## Video Script Outline

1. Motivation — "embeddings gave us meaning, now we need memory across a sequence"
2. Vanilla RNN recurrence + unrolling diagram
3. BPTT and the sequential-dependency problem
4. Vanishing gradients — build the geometric-series intuition
5. LSTM gates, cell state, "why additive beats multiplicative" for gradient flow
6. GRU as the leaner cousin
7. Walkthrough of `example.py` — measure gradient decay in RNN vs. LSTM directly
8. Recap: gradients are fixed, but recurrence itself remains -> preview attention

## Further Reading

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (GRU)
- Christopher Olah, *Understanding LSTM Networks* (colah.github.io) — the canonical visual explainer
- Pascanu, Mikolov, Bengio (2013), *On the difficulty of training Recurrent Neural Networks* (vanishing/exploding gradients, gradient clipping)

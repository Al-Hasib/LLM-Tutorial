# Sequence-to-Sequence and Attention

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `04-Seq2Seq-and-Attention`

## Why this matters

This is the single most important lesson in this phase: **attention** — the mechanism this entire course's models are named after and built around — was invented right here, as a patch for a specific, narrow failure of RNN-based translation models. Once you see the problem it was designed to solve, self-attention in [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/02-Self-Attention-and-Multi-Head-Attention/README.md) stops looking like a mysterious new idea and starts looking like the obvious generalization of what you're about to build here by hand.

## What this lesson covers

- The Sequence-to-Sequence (Seq2Seq) architecture: encoder + decoder
- The fixed-context bottleneck problem
- Attention: letting the decoder look back at every encoder state
- The attention formula: scores → softmax → weighted context
- Additive (Bahdanau) vs. multiplicative (Luong / dot-product) attention
- Why this is already "query, key, value" in disguise

## 1. Sequence-to-Sequence (Seq2Seq)

Introduced for machine translation (Sutskever et al., 2014), a Seq2Seq model is two RNNs chained together:

- **Encoder**: reads the entire source sentence, one word at a time, and compresses it into a single fixed-size vector — its final hidden state. This is often called the "context vector" or "thought vector."
- **Decoder**: another RNN, initialized with that context vector, that generates the target sentence one word at a time, feeding each generated word back in as its own next input.

```
source sentence -> [Encoder RNN] -> context vector -> [Decoder RNN] -> target sentence
```

## 2. The fixed-context bottleneck

Here's the problem: **the entire meaning of an arbitrarily long source sentence has to be squeezed through one fixed-size vector.** For a 5-word sentence this is barely noticeable; for a 40-word sentence, information from the beginning of the sentence has usually already been diluted or overwritten by the time the encoder reaches the end — the same vanishing-signal issue from [RNNs, LSTMs and GRUs](../03-RNN-LSTM-GRU/README.md), now compounded by forcing *everything* through a single vector. Translation quality measurably degrades as source sentences get longer, and this bottleneck is exactly why.

## 3. The fix: attention

Bahdanau et al. (2014) proposed a direct fix: **stop compressing the whole sentence into one vector.** Instead, keep *every* encoder hidden state around, and let the decoder, at each generation step, look back across all of them and decide which ones are relevant right now.

Concretely, at each decoder step `t`, with decoder state `h_t^{dec}` and encoder hidden states `h_1^{enc}, ..., h_n^{enc}`:

```
score(h_t^{dec}, h_i^{enc}) = how relevant is encoder position i right now?
α_{t,i} = softmax_i( score(h_t^{dec}, h_i^{enc}) )        # attention weights, sum to 1
context_t = Σ_i α_{t,i} · h_i^{enc}                        # weighted sum = attention output
```

That `context_t` — a custom blend of the whole source sentence, recomputed fresh at every decoding step — is then combined with the decoder state to produce the next output word. **This is exactly a differentiable, weighted lookup**: the softmax turns raw scores into a probability-like distribution ([Phase 00 §4](../../Phase-00-Prerequisites/01-Python-and-Math-Refresher/README.md#4-probability)), and the "answer" is a weighted average, not a hard pick of one single position.

## 4. Additive vs. multiplicative attention

Two popular ways to compute the score:

- **Additive / Bahdanau attention**: `score = vᵀ tanh(W₁ h^{dec} + W₂ h^{enc})` — a small feedforward network learns the scoring function.
- **Multiplicative / Luong (dot-product) attention**: `score = h^{dec} · h^{enc}` (or `h^{dec}ᵀ W h^{enc}`) — just a dot product, optionally with a learned matrix in between.

Dot-product attention is cheaper (one matrix multiply instead of a small neural net per pair) and, once scaled properly, is *exactly* the mechanism [Transformers](../05-Intro-to-Transformers/README.md) standardize on for every attention computation in the model.

## 5. This is already Query/Key/Value

Relabel the pieces and the connection to Transformer terminology becomes explicit:

| Seq2Seq attention | Transformer terminology |
|---|---|
| decoder state `h_t^{dec}` | **query** — "what am I looking for right now?" |
| each encoder state `h_i^{enc}` | **key** — "here's what I contain, compare against me" |
| each encoder state `h_i^{enc}` (again) | **value** — "here's what to actually retrieve if I'm relevant" |
| `softmax(score) · encoder states` | the attention output |

The only real generalization Transformers make (next lesson, and in full in Phase 02) is: **stop restricting attention to "decoder looking at encoder."** Let *every* position in a sequence attend to *every other* position, including within the same sequence — that's **self**-attention, and it's what finally let the field drop recurrence entirely.

## Video Script Outline

1. Motivation — "translation models had one specific bug; the fix became the most important idea in modern AI"
2. Seq2Seq architecture diagram: encoder squeeze, decoder generate
3. The bottleneck, made concrete with a long sentence example
4. Attention: scores -> softmax -> weighted context, step by step
5. Additive vs. multiplicative scoring
6. Walkthrough of `example.py` — attention as a "soft lookup" on a toy copy task, and a bottleneck-vs-attention comparison
7. Recap: relabel as Q/K/V -> preview self-attention and the end of recurrence

## Further Reading

- Sutskever, Vinyals, Le (2014), *Sequence to Sequence Learning with Neural Networks*
- Bahdanau, Cho, Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong, Pham, Manning (2015), *Effective Approaches to Attention-based Neural Machine Translation*
- Jay Alammar, *Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)* (blog)

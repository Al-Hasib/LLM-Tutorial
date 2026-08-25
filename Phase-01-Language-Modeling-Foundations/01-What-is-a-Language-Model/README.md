# What is a Language Model

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `01-What-is-a-Language-Model`

## Why this matters

Everything in this entire course — a 7B-parameter chatbot included — is a language model, and every language model is doing exactly one thing: assigning a probability to a sequence of words (or tokens), and using that to guess what comes next. Before touching neural networks, it's worth seeing this idea in its simplest possible form — counting words in a corpus — because the strengths and, more importantly, the *failures* of that simple form are precisely what motivate every neural technique in the rest of this course.

## What this lesson covers

- The formal definition of a language model
- N-gram models and the Markov assumption
- Estimating probabilities from a corpus by counting
- The zero-probability problem and smoothing
- Perplexity: how we measure whether a language model is any good
- Generating text by sampling from an n-gram model
- Why n-grams break down, and what that implies for everything that follows

## 1. What a language model actually is

A language model assigns a probability to any sequence of words `w₁, w₂, ..., w_T`. Using the chain rule of probability, that joint probability factors into a product of next-word predictions:

```
P(w₁, w₂, ..., w_T) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(w_T|w₁,...,w_{T-1})
```

That's it — that's the entire idea. Train a model to be good at `P(next word | everything so far)`, and you can: score how likely any sentence is, and generate new text by repeatedly sampling from that next-word distribution. Every architecture in this course (n-grams, RNNs, Transformers) is a different way of approximating `P(w_t | w₁, ..., w_{t-1})`.

## 2. N-gram models and the Markov assumption

Conditioning on the *entire* preceding history is intractable to estimate from data — there are essentially infinite possible histories. The **Markov assumption** simplifies this: assume the next word only depends on the previous `n-1` words.

- **Bigram model** (n=2): `P(w_t | w₁,...,w_{t-1}) ≈ P(w_t | w_{t-1})`
- **Trigram model** (n=3): `P(w_t | w₁,...,w_{t-1}) ≈ P(w_t | w_{t-2}, w_{t-1})`

## 3. Estimating probabilities by counting (MLE)

Given a training corpus, the maximum-likelihood estimate for a bigram probability is just relative frequency:

```
P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
```

Count how often the pair occurred, divide by how often the first word occurred at all.

## 4. The zero-probability problem and smoothing

If a word pair never appeared in training, its count is `0`, so the model assigns it probability `0` — and multiplying anything by `0` makes the *entire sentence* impossible, even if only one bigram was unseen. This is far too brittle for any real text. **Laplace (add-one) smoothing** fixes it by pretending every possible pair occurred at least once:

```
P(w_t | w_{t-1}) = (count(w_{t-1}, w_t) + 1) / (count(w_{t-1}) + V)
```

where `V` is the vocabulary size. Cruder than modern smoothing methods (Kneser-Ney, etc.), but the same core idea — never let the model assign exactly zero probability to something it simply hasn't seen — reappears constantly, e.g. in label smoothing during LLM training.

## 5. Perplexity

Perplexity is the standard way to score a language model, defined as the exponentiated average negative log-likelihood per token:

```
PPL = exp( -(1/T) Σ log P(w_t | w_{t-1}) )
```

Intuitively: perplexity is "the average number of equally-likely choices the model thinks it's choosing between at each step." A perfect model that always assigns probability 1 to the correct next word has `PPL = 1`. A model that's totally lost, uniformly guessing among `V` vocabulary words, has `PPL = V`. **Lower is better.** This exact metric (just computed with a neural model's probabilities instead of n-gram counts) is still reported for every LLM you'll see in [Phase 08: Evaluation Metrics](../../Phase-08-Evaluation-of-LLMs/01-Evaluation-Metrics/README.md).

## 6. Generating text from an n-gram model

Once you have `P(w_t | w_{t-1})` for every word, you can generate text: start with a seed word, sample the next word from its conditional distribution, then repeat — feeding each sampled word back in as the new context. This sample-then-feed-back loop is *exactly* how every LLM in this course generates text at inference time; only the probability distribution being sampled from gets more sophisticated.

## 7. Why n-grams break down

- **Sparsity**: most reasonable trigrams never appear in even a huge corpus, so counts are unreliable — going to 4-grams or 5-grams to capture more context makes sparsity catastrophically worse.
- **No generalization**: the model has no idea that "cat" and "dog" are similar animals — `P(w | "the cat")` and `P(w | "the dog")` are estimated completely independently, however similar cats and dogs are semantically. This is the exact same one-hot-encoding blind spot from [Phase 00: Introduction to NLP](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#5-one-hot-encoding-and-its-limits).
- **Fixed, tiny context window**: real language depends on words far outside a 2-3 word window, which n-grams structurally cannot see.

These three failures are precisely what motivate the rest of this phase: [Word Embeddings](../02-Word-Embeddings/README.md) fix the "no generalization" problem, and [RNNs](../03-RNN-LSTM-GRU/README.md) and eventually [Transformers](../05-Intro-to-Transformers/README.md) fix the "fixed, tiny context" problem.

## Video Script Outline

1. Motivation — "a language model is just next-word prediction; here's the simplest version that could possibly work"
2. Chain rule of probability -> Markov assumption -> bigram/trigram models
3. Counting-based estimation on a real example, then hit the zero-probability wall
4. Smoothing fixes it, perplexity measures it
5. Walkthrough of `example.py` — train, score, and generate from a bigram model
6. Recap: three specific failure modes -> preview exactly which future lesson fixes each one

## Further Reading

- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 (N-gram Language Models)
- Chen & Goodman (1999), *An Empirical Study of Smoothing Techniques for Language Modeling*
- Shannon (1948), *A Mathematical Theory of Communication* — the origin of using entropy/perplexity-like quantities for language

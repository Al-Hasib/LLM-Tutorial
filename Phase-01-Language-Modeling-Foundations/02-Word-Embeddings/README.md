# Word Embeddings

**Phase:** [Language Modeling Foundations](../README.md) · **Topic folder:** `02-Word-Embeddings`

## Why this matters

The previous lesson ended with n-grams' fatal flaw: they treat "cat" and "dog" as two totally unrelated symbols, so nothing learned about one transfers to the other. Word embeddings are the fix — dense vectors, learned from raw text, positioned so that words used in similar contexts end up geometrically close together. This single idea (turn a discrete token into a point in continuous space that captures meaning) is the foundation everything downstream is built on: every Transformer's first layer is still, fundamentally, an embedding lookup.

## What this lesson covers

- Why dense embeddings beat one-hot vectors (recap + extension of [Phase 00 §5-6](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#5-one-hot-encoding-and-its-limits))
- Word2Vec: CBOW and Skip-gram
- Negative sampling (why you can't naively softmax over the whole vocabulary)
- GloVe: global co-occurrence statistics
- FastText: subword embeddings and the out-of-vocabulary problem
- Embedding arithmetic and its limits
- Why *static* embeddings still aren't enough (the bridge to attention and Transformers)

## 1. From one-hot to dense vectors

A one-hot vector for a 50,000-word vocabulary is 50,000 dimensions, almost all zero, and every pair of distinct words has cosine similarity exactly `0` (see the [demo in Phase 00](../../Phase-00-Prerequisites/03-Intro-to-NLP/example.py)). An **embedding** replaces that with a small dense vector (e.g. 100-300 dimensions) where distance and direction are meaningful. The [distributional hypothesis](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#6-the-distributional-hypothesis) tells us how to learn such a thing without any manual labeling: **train a model to predict context from words (or words from context), and the internal representation it's forced to build turns out to capture meaning as a side effect.**

## 2. Word2Vec: CBOW and Skip-gram

Two mirror-image training objectives, both from Mikolov et al. (2013):

- **CBOW (Continuous Bag-of-Words)**: predict the center word from its surrounding context words.
- **Skip-gram**: predict each surrounding context word from the center word (empirically works better on smaller datasets / rare words, and is what we implement below).

Concretely, for a sentence and a window size `w`, skip-gram builds `(center, context)` pairs — e.g. for `"the cat sat on the mat"` with window 2, the center word `"sat"` produces training pairs `(sat, cat)`, `(sat, on)`, `(sat, the)`, `(sat, mat)`. The model learns two embedding matrices — one for a word as a "center," one for a word as a "context" — such that `center · context` is large for pairs that co-occur and small otherwise.

## 3. Negative sampling

Turning `center · context` similarity into a probability naively requires a softmax over the *entire vocabulary* for every single training pair — computationally brutal at real vocabulary sizes. **Negative sampling** reframes training as simple binary classification instead: "is `(center, context)` a real pair from the data, or a randomly sampled fake one?" For every real pair, sample a handful of random "negative" context words and push their dot product down while pushing the real pair's dot product up. This turns an O(vocab size) softmax into an O(k) binary decision per step, where `k` is small (5-20).

## 4. GloVe: global co-occurrence statistics

Word2Vec only ever looks at local windows, one pair at a time. **GloVe** (Pennington et al., 2014) instead builds a full word-word co-occurrence matrix over the entire corpus first, then factorizes it so that the dot product of two word vectors approximates the log of their co-occurrence count. Roughly: Word2Vec learns from many small local views; GloVe learns from one global statistic. In practice, both produce embeddings of comparable quality.

## 5. FastText: subwords fix out-of-vocabulary words

Both Word2Vec and GloVe assign one opaque vector per whole word — an unseen word at test time (a typo, a new product name, a rare inflection) simply has no embedding. **FastText** (Bojanowski et al., 2017) represents each word as the sum of its **character n-gram** embeddings (e.g. `"running"` → `<ru, run, unn, nni, nin, ing, ng>` plus the whole word). A never-before-seen word can still be embedded by summing whatever subword pieces it shares with known words — and shared morphology ("run", "runs", "running", "runner") naturally pulls related words' vectors closer together. This subword idea reappears, generalized, as the **byte-pair encoding tokenizers** covered in [Phase 02: Tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md).

## 6. Embedding arithmetic — and its limits

The famous demo: `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`. Directions in embedding space end up encoding relationships (gender, tense, capital-of-country), not just proximity. This is a real, reproducible phenomenon on embeddings trained on large corpora — but it's also fragile and easy to overstate on small corpora, as `example.py` below will show honestly.

## 7. Why static embeddings still aren't enough

Word2Vec/GloVe/FastText give **one fixed vector per word**, regardless of context — "bank" gets the same embedding in "river bank" and "bank account." That's a real limitation: meaning is context-dependent, and these methods can't represent that. Fixing this — making a word's representation depend on the *specific sentence* it appears in — is exactly what self-attention in Transformers achieves, which is why the field moved from these static embeddings to the **contextual embeddings** produced by models like BERT and GPT. That's where [Introduction to Transformers](../05-Intro-to-Transformers/README.md) and [Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) pick up.

## Video Script Outline

1. Motivation — recap one-hot's blind spot, "what if position in space could mean something?"
2. CBOW vs. skip-gram, with the sliding-window pair-extraction example on screen
3. Negative sampling — why you can't just softmax 50,000 words every step
4. GloVe and FastText in contrast to Word2Vec
5. Walkthrough of `example.py` — train tiny skip-gram embeddings, inspect nearest neighbors
6. Recap: dense vectors solve "no generalization," but they're still *static* — preview contextual embeddings

## Further Reading

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* (Word2Vec)
- Mikolov et al. (2013), *Distributed Representations of Words and Phrases and their Compositionality* (negative sampling)
- Pennington, Socher, Manning (2014), *GloVe: Global Vectors for Word Representation*
- Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (FastText)

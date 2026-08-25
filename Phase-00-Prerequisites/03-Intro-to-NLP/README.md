# Introduction to NLP

**Phase:** [Prerequisites](../README.md) · **Topic folder:** `03-Intro-to-NLP`

## Why this matters

Neural networks only understand numbers. Before any language model — from a toy bigram model to GPT-4-scale systems — can do anything with text, that text has to become vectors. This lesson covers how NLP represented text *before* modern subword tokenization and learned embeddings took over (those get their own deep dives in [Phase 01: Word Embeddings](../../Phase-01-Language-Modeling-Foundations/02-Word-Embeddings/README.md) and [Phase 02: Tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md)). Understanding the classic approaches first makes it obvious *why* embeddings were such a breakthrough.

## What this lesson covers

- Text preprocessing basics (and why LLMs mostly skip most of it today)
- Tokenization at the word level
- Bag-of-Words (BoW)
- TF-IDF
- One-hot encoding and its limitations
- The distributional hypothesis (the idea that motivates every embedding method later in the course)
- Cosine similarity for comparing text vectors
- N-grams

## 1. Text preprocessing

Classic NLP pipelines typically applied:

- **Lowercasing** — "Cat" and "cat" become the same token
- **Punctuation/stopword removal** — drop low-information words ("the", "is", "a")
- **Stemming/lemmatization** — reduce "running"/"ran"/"runs" to a common root ("run")

Modern LLMs mostly **skip** this pipeline: subword tokenizers (Phase 02) handle casing and morphology implicitly by learning frequent subword units directly from raw text, and stopwords carry useful signal for a large enough model. It's still essential to know this pipeline, both historically and because it's still used in lightweight/classical NLP systems (search, spam filters, etc.).

## 2. Word-level tokenization

The simplest possible tokenizer: split on whitespace/punctuation.

```
"The cat sat on the mat." -> ["the", "cat", "sat", "on", "the", "mat", "."]
```

Problems this creates: huge vocabularies, no way to handle unseen words ("out-of-vocabulary"), and no shared structure between related words ("run", "running", "runner" are three unrelated tokens). Subword tokenization (Phase 02) exists specifically to solve these problems.

## 3. Bag-of-Words (BoW)

Represent a document as a vector counting how often each vocabulary word appears, ignoring order and grammar entirely:

```
vocab = ["the", "cat", "sat", "mat", "dog"]
"the cat sat on the mat" -> [2, 1, 1, 1, 0]
```

Simple and surprisingly effective for tasks like spam detection, but throws away word order completely — "dog bites man" and "man bites dog" get very different meanings but could share similar BoW vectors.

## 4. TF-IDF

Term Frequency–Inverse Document Frequency down-weights words that appear in *every* document (uninformative) and up-weights words that are frequent in one document but rare across the corpus (informative):

```
TF(t, d)  = count of term t in document d
IDF(t)    = log( N / (1 + document_count(t)) )     # N = total number of documents
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

A word like "the" appears in every document → `IDF ≈ 0` → contributes almost nothing. A rare, topic-specific word gets a high IDF and dominates the vector for documents that contain it.

## 5. One-hot encoding and its limits

Representing a word as a one-hot vector — all zeros except a single 1 at that word's index — makes the vocabulary size the vector size (potentially hundreds of thousands of dimensions), and critically, **the dot product between any two distinct one-hot vectors is always 0**. There's no notion that "cat" and "dog" are more related than "cat" and "astronomy." This is the exact limitation dense embeddings solve.

## 6. The distributional hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth, 1957

Words that occur in similar contexts tend to have similar meanings. This single idea is the foundation of every embedding method covered later: [Word2Vec, GloVe, and FastText](../../Phase-01-Language-Modeling-Foundations/02-Word-Embeddings/README.md) all learn vectors by modeling word co-occurrence, directly operationalizing this hypothesis.

## 7. Cosine similarity

Once text is a vector, the standard way to measure "how similar are these two pieces of text" is the cosine of the angle between their vectors, not raw distance (which is sensitive to document length):

```
cos_sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Ranges from -1 (opposite) to 1 (identical direction); 0 means unrelated. This is the same operation embedding-based search, RAG retrieval ([Phase 08](../../Phase-08-RAG-and-Agents/01-Embeddings-and-Vector-Databases/README.md)), and attention scores all build on.

## 8. N-grams

A simple fix for BoW's "no word order" problem: count contiguous sequences of `n` words instead of single words. "the cat sat" as bigrams: `["the cat", "cat sat"]`. Captures a little local order at the cost of a much larger, sparser vocabulary.

## Video Script Outline

1. Motivation — "computers only see numbers; here's the pre-embedding way to get there"
2. Preprocessing pipeline walkthrough
3. BoW and TF-IDF derivations with a tiny 3-document example
4. One-hot's dot-product problem, live in code
5. The distributional hypothesis — the bridge to next phase's Word2Vec
6. Walkthrough of `example.py`
7. Recap + pointer to PyTorch Fundamentals next

## Further Reading

- Jurafsky & Martin, *Speech and Language Processing*, Ch. 2 (regex/tokenization) & Ch. 6 (vector semantics)
- scikit-learn docs: `CountVectorizer`, `TfidfVectorizer`
- Firth, J.R. (1957), *A synopsis of linguistic theory* — origin of the distributional hypothesis

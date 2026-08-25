# Evaluation Metrics

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `01-Evaluation-Metrics`

## Why this matters

[Phase 01 Lesson 1](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md#5-perplexity) introduced perplexity as "the" way to score a language model, computed there from raw bigram counts. Every LLM you'll ever read a paper about still reports perplexity — just computed from a neural network's probabilities instead of a count table. But perplexity only measures how well a model predicts held-out *training-distribution* text; it says nothing about whether a *generated* piece of text (a translation, a summary, an answer) is actually good. That gap is what this lesson fills: the classic automatic metrics — BLEU, ROUGE, exact match, token F1 — built specifically to score *generated output* against a reference, by counting overlapping words. They are cheap, deterministic, and reproducible, which is exactly why they dominated NLP evaluation for two decades — and exactly why, as you'll see by the end of this lesson, they can be fooled by a perfectly good answer that simply uses different words. That honest limitation is the whole reason [Lesson 3: LLM-as-a-Judge](../03-LLM-as-a-Judge/README.md) exists: once you see *why* n-gram overlap breaks on paraphrase, using a second LLM to judge meaning instead of surface overlap stops looking like an exotic idea and starts looking necessary.

## What this lesson covers

- Perplexity, recapped for a neural model instead of an n-gram table
- BLEU: n-gram precision with a brevity penalty, the classic machine-translation metric
- ROUGE-N and ROUGE-L: recall-oriented overlap metrics, the classic summarization metrics
- Exact Match and token-level F1: the standard extractive-QA metrics
- The shared honest weakness of every overlap-based metric, demonstrated concretely

## 1. Perplexity, recapped

Recall the definition from Phase 01:

```
PPL = exp( -(1/T) * sum_t log P(w_t | w_1, ..., w_{t-1}) )
```

Nothing changes conceptually for an LLM — `P(w_t | ...)` is now produced by a Transformer's softmax instead of a smoothed count ratio, and `T` is measured in *subword tokens* ([Phase 02 Lesson 1](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md)) rather than whole words. Perplexity is useful and cheap (it needs no reference *output*, just held-out text the model wasn't trained on) but it only scores the model's own next-token distribution over naturally occurring text. It cannot score a *generated* translation, summary, or answer against a *reference* — for that you need a metric that compares two pieces of text directly. That's what the rest of this lesson is about.

## 2. BLEU (machine translation)

BLEU (**BiLingual Evaluation Understudy**, Papineni et al., 2002) scores a candidate translation by how much of its n-gram content also appears in one or more human reference translations — it's fundamentally a **precision** metric: "of the n-grams the model produced, how many were actually right?"

**Modified n-gram precision.** Plain precision can be gamed by repeating a single correct word forever (e.g. candidate `"the the the the"` against reference `"the cat sat down"` scores 100% unigram precision). BLEU fixes this by *clipping* each n-gram's count in the candidate to its maximum count in any single reference:

```
p_n = ( sum over n-grams in candidate of min(count_candidate(ngram), count_reference(ngram)) )
      / ( total count of n-grams in candidate )
```

**Brevity penalty.** Precision alone also rewards being short — a candidate that outputs only "the" gets perfect precision on that one word. BLEU counteracts this with a **brevity penalty (BP)** that punishes candidates shorter than the reference:

```
BP = 1                       if c > r
BP = exp(1 - r/c)            if c <= r
```

where `c` is the candidate length and `r` is the (closest) reference length. A candidate exactly as long as the reference incurs no penalty; a candidate half the reference's length gets `BP = exp(1 - 2) = exp(-1) ~= 0.37`.

**Full BLEU score.** Combine precisions for n-grams of several orders (usually n=1..4) as a weighted geometric mean, then apply the brevity penalty:

```
BLEU = BP * exp( sum_{n=1}^{N} w_n * log(p_n) )
```

with `w_n = 1/N` typically (equal weight to unigrams through 4-grams). BLEU ranges from 0 to 1 (often reported x100).

## 3. ROUGE (summarization)

ROUGE (**Recall-Oriented Understudy for Gisting Evaluation**, Lin, 2004) flips BLEU's emphasis: since a good summary should *capture* the reference's content, ROUGE is fundamentally a **recall** metric: "of the n-grams in the reference, how many did the candidate reproduce?" (Modern usage typically reports an F-measure combining both directions, but the original motivation is recall.)

**ROUGE-N** (n-gram recall):

```
ROUGE-N = ( sum over n-grams in reference of min(count_candidate(ngram), count_reference(ngram)) )
          / ( total count of n-grams in reference )
```

Same clipped-overlap numerator as BLEU's precision, but divided by the *reference's* n-gram count instead of the candidate's — this is what makes it recall rather than precision.

**ROUGE-L** (longest common subsequence): instead of fixed-length n-grams, find the longest common subsequence (LCS) between candidate and reference — a common subsequence need not be contiguous, so it tolerates word insertions/reorderings that would break an n-gram match. Given `LCS(X, Y)`, the length of the longest common subsequence between candidate `X` and reference `Y`:

```
R_lcs = LCS(X, Y) / len(Y)             # recall
P_lcs = LCS(X, Y) / len(X)             # precision
F_lcs = (1 + beta^2) * R_lcs * P_lcs / (R_lcs + beta^2 * P_lcs)
```

`beta` controls the recall/precision trade-off (`beta` large favors recall, matching ROUGE's original recall-oriented spirit; `beta=1` gives the harmonic mean, an ordinary F1).

## 4. Exact Match and token F1 (extractive QA)

For extractive QA (the answer is a literal span copied from a passage, as in SQuAD), the metrics are simpler and stricter:

**Exact Match (EM)**: after light normalization (lowercase, strip punctuation/articles), 1 if the predicted string is character-for-character identical to a reference answer, else 0. Averaged over a dataset, EM is just accuracy.

**Token F1**: treats the prediction and the (normalized, whitespace-tokenized) reference answer as bags of tokens and computes:

```
overlap    = number of tokens shared between prediction and reference (as a multiset intersection)
precision  = overlap / len(prediction tokens)
recall     = overlap / len(reference tokens)
F1         = 2 * precision * recall / (precision + recall)
```

Token F1 gives partial credit EM cannot: predicting `"the Eiffel Tower"` against a reference of `"Eiffel Tower"` scores `EM=0` but `F1` well above zero, since 2 of the prediction's 3 tokens are correct.

## 5. The shared weakness: surface overlap is not meaning

Every metric above — BLEU, ROUGE, EM, token F1 — measures **lexical overlap**: how many of the same words/n-grams appear in the same places. None of them has any notion of *meaning*. Two consequences follow directly, and both are demonstrated with real numbers in `example.py`:

- **Valid paraphrases are punished.** A candidate that says the same thing in different words (correct synonyms, reordered clauses, a different but equally valid phrasing) can score *near zero* on BLEU/ROUGE even though a human reader would call it a perfect answer — because there is no n-gram overlap to reward.
- **Overlap can be gamed.** A candidate that copies large verbatim chunks of the reference (or the source passage) scores artificially high regardless of whether it is coherent, faithful, or even correct as a complete answer — optimizing directly against these metrics (e.g. during model selection or as a training reward) can push a system toward parroting reference-like phrasing rather than genuinely improving quality.

This is not a minor footnote — it is the primary reason serious LLM evaluation moved toward semantic-embedding-based metrics, model-based scoring, and ultimately [LLM-as-a-Judge](../03-LLM-as-a-Judge/README.md) for open-ended generation, while BLEU/ROUGE/EM/F1 remain useful mainly as cheap, reproducible sanity checks on narrower, more constrained tasks (translation with tight reference phrasing, extractive QA with an unambiguous gold span) where surface overlap and semantic correctness are more tightly coupled.

## Video Script Outline

1. Motivation — perplexity scores the model's own distribution, but can't score a generated answer against a reference; what can?
2. BLEU: n-gram precision, why clipping is needed, the brevity penalty and why it exists
3. ROUGE: flip to recall, ROUGE-N vs. ROUGE-L (LCS) and why LCS tolerates reordering
4. Exact Match and token F1 for extractive QA, and why F1 gives partial credit EM can't
5. Walkthrough of `example.py` — implement all of these from scratch, verify them on toy sentence pairs
6. The honest demo: a good paraphrase scores badly on BLEU/ROUGE despite being correct
7. Perplexity vs. these metrics, side by side, on the same toy model/sentences
8. Recap: why this gap in surface-overlap metrics directly motivates LLM-as-a-Judge (next lesson)

## Further Reading

- Papineni, Roukos, Ward, Zhu (2002), *BLEU: a Method for Automatic Evaluation of Machine Translation*
- Lin (2004), *ROUGE: A Package for Automatic Evaluation of Summaries*
- Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Comprehension of Text* (origin of the EM/F1 QA evaluation convention)
- Callison-Burch, Osborne, Koehn (2006), *Re-evaluating the Role of BLEU in Machine Translation Research* (an early, influential critique of BLEU's weaknesses)

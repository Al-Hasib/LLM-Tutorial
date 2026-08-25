# Pretraining Data Pipeline

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `01-Pretraining-Data-Pipeline`

## Why this matters

Every lesson before this phase assumed clean, ready-to-use text already existed — [Phase 01's bigram model](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md), [BPE tokenization](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md), and [the mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) all trained on a small, hand-picked corpus. Real pretraining corpora are assembled from hundreds of terabytes of raw internet text, and getting that raw text into a form worth spending [Scaling-Laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md)-worthy compute on is itself a massive engineering effort — arguably the least glamorous, most consequential part of building an LLM. Bad data doesn't just waste compute; it actively teaches the model bad habits (parroting boilerplate, memorizing exact strings, or worse, memorizing the very benchmark questions used to evaluate it). This lesson is the first of four that together explain everything that happens *before* [Lesson 5's capstone training run](../05-Pretraining-a-Small-LLM-From-Scratch/README.md).

## What this lesson covers

- Where pretraining text actually comes from (Common Crawl and other sources)
- Extracting usable text from raw, messy HTML
- Quality filtering: heuristic rules and trained classifiers
- Deduplication: exact (hashing) and near-duplicate (MinHash/LSH) detection, and why duplicates are actively harmful
- Data mixing: why the *ratio* of sources matters as much as the total token count

## 1. Where the text comes from

No single source has enough naturally-occurring high-quality text to pretrain a modern LLM, so real pipelines blend several:

- **Common Crawl** — a public, continuously updated snapshot of a large fraction of the crawlable web (petabytes of raw HTML/text), the backbone of nearly every large pretraining corpus (GPT-3's dataset, The Pile, RefinedWeb, FineWeb, etc.). It's enormous but extremely noisy — spam, auto-generated pages, boilerplate, and non-language content dominate a raw crawl.
- **Curated high-quality sources** — Wikipedia, books (Project Gutenberg, Books3-style corpora), academic papers (arXiv, PubMed), and code (GitHub). These are far smaller in raw byte count than Common Crawl but disproportionately valuable per token: cleaner prose, denser factual content, and (for code) a completely different, highly structured modality that measurably improves reasoning on non-code tasks too.
- **Forums and social/QA data** (Reddit, StackExchange) — conversational and question-answering structure that web prose and books don't provide.

The raw ratio of "how much of each exists on the internet" is *not* the ratio you want to train on — see §5.

## 2. Text extraction and cleaning

A raw Common Crawl record is an HTML page, not clean prose. Before any of the later stages can run, you need to pull out just the article/content text:

- **Boilerplate removal** — strip navigation menus, headers/footers, cookie banners, ads, and sidebar links, keeping only the main content block. Tools like `trafilatura` and `jusText` do this with heuristics over the HTML DOM (text density, tag structure, link density per block).
- **Encoding and language normalization** — fix character-encoding errors, normalize Unicode, and run language identification (e.g. `fastText`'s language classifier) to route documents into per-language corpora or discard languages you don't want.
- **Markup stripping** — remove residual HTML tags, JavaScript, and CSS that boilerplate removal missed.

This step alone typically throws away the large majority of a raw crawl's bytes before quality filtering even begins.

## 3. Quality filtering

Even after boilerplate removal, most remaining documents are still low-value: SEO spam, auto-generated product listings, keyword-stuffed pages, or just very short fragments. Two complementary filtering strategies are used together:

**Heuristic filters** — cheap, rule-based checks applied to every document, for example (following the recipes used by CCNet, Gopher/MassiveText, and RefinedWeb):

- Reject documents below a minimum word/token count (too short to carry useful signal)
- Reject documents with an abnormally high symbol-to-word ratio (excess punctuation, emoji, or markup residue suggests spam or extraction failure)
- Reject documents with excessive line- or n-gram-level repetition (templated pages, auto-generated boilerplate that survived extraction)
- Require a minimum fraction of stop-words or a plausible word-length distribution (a crude but effective "does this look like real prose" check)

**Classifier-based filtering** — train a lightweight classifier (historically a `fastText` linear classifier, since it's cheap enough to run over an entire crawl) to distinguish "high quality" text from "typical web text," using known-good sources (Wikipedia, curated books, OpenWebText-style human-curated links) as positive examples and random crawl documents as negative examples. Every crawl document then gets a quality score, and the pipeline keeps only documents above a threshold. This is strictly more powerful than heuristics because it can pick up on subtler stylistic signals no hand-written rule captures — at the cost of needing labeled reference data and being only as good as the definition of "high quality" baked into that reference set.

## 4. Deduplication

The web is extraordinarily repetitive: the same article gets syndicated across dozens of news mirrors, boilerplate legal text (privacy policies, license headers) appears near-verbatim on millions of pages, and forum threads get quoted-and-requoted. Left in the training set, duplicates cause three distinct problems:

- **Memorization** — a model sees the exact same string many times across training, making it far more likely to memorize and later regurgitate that string verbatim (a real privacy and copyright concern) instead of generalizing.
- **Wasted compute** — every duplicate token still costs a forward/backward pass; training on the same content ten times provides far less than ten times the learning signal.
- **Benchmark contamination** — if a duplicate (or near-duplicate) of a test benchmark's questions leaks into the training set via some crawled copy of that benchmark, evaluation scores become meaningless (the model didn't *solve* the problem, it *memorized the answer key*).

Two dedup strategies operate at different granularities:

**Exact deduplication** — hash each document (or each line, for finer granularity) with a fast hash function (e.g. SHA-1) and drop exact hash collisions. Cheap and perfectly precise, but only catches byte-identical duplicates — it misses the vastly more common case of *near*-duplicates: the same article with a different ad inserted, a re-published version with one paragraph edited, or a template page with only a product name changed.

**Near-duplicate detection via MinHash + LSH** — to catch near-duplicates, represent each document as a set of overlapping word (or character) **shingles** (n-grams), and estimate the **Jaccard similarity** between two documents' shingle sets:

```
Jaccard(A, B) = |A intersect B| / |A union B|
```

Computing this exactly for every pair of documents in a billion-document corpus is intractable (quadratic in corpus size). **MinHash** makes it tractable: apply many independent hash functions to a document's shingle set, keep only the *minimum* hashed value per hash function as that document's "signature" for that function, and repeat for `k` independent hash functions to get a `k`-element signature vector. The key mathematical property that makes this work:

```
P( minhash_h(A) == minhash_h(B) ) = Jaccard(A, B)
```

for a single random hash function `h`. So the *fraction of matching positions* between two documents' `k`-element MinHash signatures is an unbiased estimator of their true Jaccard similarity — computed from two short signatures instead of two full shingle sets. In production this is further sped up with **Locality-Sensitive Hashing (LSH)**: bucket documents by bands of their MinHash signature so only documents that already agree on a band are ever compared, avoiding the full quadratic pairwise comparison entirely. `example.py` implements MinHash signatures and Jaccard estimation directly, and shows exact-hash dedup failing to catch a near-duplicate pair that MinHash correctly flags.

## 5. Data mixing

Once you have a large pool of cleaned, deduplicated documents from several sources, the final decision is: **what fraction of the training mixture should come from each source?** This matters as much as total token count, for two reasons:

- **Value density differs wildly by source.** A token of Wikipedia or a peer-reviewed paper carries more useful signal, on average, than a token of a random forum comment — so most real pipelines *upweight* (sample more often than its raw share of the pool) high-quality curated sources like books, Wikipedia, and code, and *downweight* the much larger but noisier web-crawl portion, relative to their raw byte counts.
- **Diversity prevents narrow competence.** A model trained purely on web prose is weak at code and formal reasoning; adding a deliberate code fraction (as GPT-3, PaLM, and LLaMA all did) measurably improves performance on tasks that have nothing to do with writing code, apparently because code teaches long-range structural and logical dependencies that transfer.

Published mixtures make this concrete: GPT-3's training mix downweighted its largest raw component (a filtered Common Crawl, `~82%` of raw tokens) to about 60% of the actual training mixture by upsampling higher-quality sources (WebText2, Books1/2, Wikipedia) several times over relative to their size. The exact ratios are treated as important tunable hyperparameters in their own right, re-derived experimentally for nearly every new model family — there is no universally "correct" mixture, only one that's been validated to work well for a given model's target capabilities.

## Video Script Outline

1. Motivation — "an LLM is only as good as the pipeline that built its training set"
2. Where the raw text comes from: Common Crawl plus curated sources
3. Extraction and cleaning: turning raw HTML into plain prose
4. Heuristic vs. classifier-based quality filtering, with concrete rejection examples
5. Exact dedup (hashing) vs. near-dedup (MinHash + Jaccard), and why duplicates actively hurt
6. Data mixing: why ratios, not just volume, decide the final mixture
7. Walkthrough of `example.py` — heuristic filter, exact-hash dedup, and MinHash near-dedup on a toy document set
8. Recap + preview of Lesson 2: once you have clean data, what objective do you actually train on?

## Further Reading

- Wenzek et al. (2020), *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*
- Rae et al. (2021), *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (the MassiveText pipeline and its filtering/dedup recipe)
- Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3 — Appendix A details its data mixture and quality-classifier filtering)
- Broder (1997), *On the Resemblance and Containment of Documents* (the original MinHash paper)
- Penedo et al. (2023/2024), *The RefinedWeb Dataset* and *FineWeb* (modern, fully-documented open web-scale filtering pipelines)

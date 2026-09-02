# Tokenization

**Phase:** [Transformer Architecture Deep Dive](../README.md) · **Topic folder:** `01-Tokenization`

## Why this matters

Every lesson so far has quietly assumed text arrives as clean "words." Real tokenizers don't work that way, and the exact algorithm a model uses to chop text into tokens shapes everything downstream: vocabulary size, sequence length, how well it handles typos and rare languages, and even how it counts letters or does arithmetic. This lesson builds the subword tokenization algorithm — Byte-Pair Encoding — that every model in this course actually uses, from scratch, so the first step of every LLM's forward pass stops being a black box.

## What this lesson covers

- Word-level vs. character-level vs. subword-level tokenization
- Byte-Pair Encoding (BPE): the algorithm, step by step
- WordPiece and SentencePiece: BPE's close cousins
- Byte-level BPE (tiktoken / GPT-style): tokenizing raw bytes, no unknown-token problem ever
- Special tokens
- The vocabulary-size trade-off

## 1. Three levels of granularity

- **Word-level** (what we implicitly used through Phase 00-01): huge vocabulary, every unseen word is `<unk>`, no shared structure between "run"/"running"/"runner" — the exact failure mode from [Phase 00: Introduction to NLP §2](../../Phase-00-Prerequisites/03-Intro-to-NLP/README.md#2-word-level-tokenization).
- **Character-level**: tiny vocabulary (~100 symbols), can represent *any* string, but sequences become very long (a 10-word sentence might be 60 characters), and the model has to learn spelling before it can learn meaning.
- **Subword-level**: the practical middle ground every modern LLM uses. Common words stay whole tokens ("the", "cat"); rare/unseen words get split into meaningful pieces ("unhappiness" → "un" + "happi" + "ness"). Best of both: short sequences for common text, graceful degradation for anything novel.

## 2. Byte-Pair Encoding (BPE)

Originally a data-compression algorithm (Gage, 1994), repurposed for NLP by Sennrich et al. (2016). The training algorithm:

1. Start with a vocabulary of individual characters (or bytes). Represent every word in the training corpus as a sequence of these symbols.
2. Count every adjacent symbol pair across the whole corpus.
3. Find the **most frequent pair**, and merge it into a single new symbol, adding it to the vocabulary.
4. Repeat steps 2-3 for a fixed number of merges (this merge count is the main hyperparameter — more merges = larger vocabulary, shorter sequences).

```
Toy example, corpus = "low lower lowest":
symbols: l o w   l o w e r   l o w e s t
merge 1: most frequent pair is (l, o) -> new symbol "lo"
         lo w   lo w e r   lo w e s t
merge 2: most frequent pair is (lo, w) -> new symbol "low"
         low   low e r   low e s t
... and so on
```

The result is a vocabulary of subword units learned directly from data — no linguist hand-designed these splits, they fell out of frequency statistics.

## 3. Encoding new text

Once merges are learned (and their order recorded), encoding a new piece of text means: start from characters, and greedily apply the learned merges *in the order they were learned* wherever they apply. A word never seen during training still encodes successfully as long as its pieces (down to individual characters, in the worst case) were seen — there is **no out-of-vocabulary problem**, unlike word-level tokenization.

## 4. WordPiece and SentencePiece

- **WordPiece** (used by BERT): nearly identical to BPE, but instead of merging the *most frequent* pair, it merges the pair that most increases the training corpus's likelihood under a simple language model — a subtly different, more "information-theoretic" merge criterion.
- **SentencePiece** (used by T5, LLaMA, and many others): a tokenizer *framework* rather than a different algorithm — it can run BPE or a unigram-language-model algorithm internally, but crucially it treats the input as a raw, un-pretokenized stream (it even encodes whitespace as a regular symbol, typically `▁`), making it language-agnostic — no assumption that words are separated by spaces, which matters enormously for languages like Chinese or Japanese.

## 5. Byte-level BPE (what GPT-family models actually use)

Character-level BPE still has one weak spot: a truly novel Unicode character (an emoji, an obscure script) that never appeared in training still can't be encoded. GPT-2 (Radford et al., 2019) fixed this permanently by running BPE over **raw UTF-8 bytes** instead of Unicode characters. There are only 256 possible byte values, so **every possible string, in any language or emoji, is representable from the very first merge** — this is why GPT-family models never emit an `<unk>` token. OpenAI's `tiktoken` library is a fast implementation of exactly this scheme, and it's what every GPT-3.5/GPT-4-family model uses.

## 6. Special tokens

Real tokenizers reserve a handful of IDs for non-text signals the model needs:

| Token                | Purpose                                                   |
| -------------------- | --------------------------------------------------------- |
| `<bos>` / `<s>`  | Beginning of sequence                                     |
| `<eos>` / `</s>` | End of sequence — often what tells generation to stop    |
| `<pad>`            | Padding shorter sequences to a common length for batching |
| `<unk>`            | Unknown symbol (rare/absent with byte-level BPE)          |

## 7. The vocabulary-size trade-off

- **Larger vocabulary** → more common multi-character chunks become single tokens → shorter sequences per sentence → cheaper self-attention (remember `O(T²)` from [Phase 01: Introduction to Transformers §5](../../Phase-01-Language-Modeling-Foundations/05-Intro-to-Transformers/README.md#5-the-trade-off-quadratic-complexity)) — but a bigger embedding table and a bigger final softmax layer.
- **Smaller vocabulary** → longer sequences, cheaper embedding/output layers, but more computation spent per unit of actual text.

Real models settle somewhere in the tens of thousands (GPT-2: ~50K, many modern LLMs: 100K-250K) as an empirically-tuned balance.

## Video Script Outline

1. Motivation — "the very first thing that happens to your prompt, and it's not what you'd guess"
2. Word vs. character vs. subword, with the OOV problem made concrete
3. BPE's merge algorithm, worked by hand on a tiny corpus
4. WordPiece and SentencePiece as close cousins
5. Byte-level BPE — why GPT models never say "unknown token"
6. Walkthrough of `example.py` — train a BPE tokenizer from scratch, encode an unseen word
7. Recap: vocabulary size trade-off, and a preview that self-attention (next lesson) operates on whatever tokens come out of this stage

## Further Reading

- Sennrich, Haddow, Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* (BPE for NLP)
- Kudo & Richardson (2018), *SentencePiece: A simple and language independent subword tokenizer*
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2 — introduces byte-level BPE), Section 2.2
- OpenAI `tiktoken` library (github.com/openai/tiktoken)

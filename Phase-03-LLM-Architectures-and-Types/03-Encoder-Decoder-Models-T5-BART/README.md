# Encoder-Decoder Models: T5 and BART

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `03-Encoder-Decoder-Models-T5-BART`

## Why this matters

Lessons 1 and 2 covered the two "pure" architectures: decoder-only (generation, no bidirectional input understanding) and encoder-only (bidirectional understanding, no generation at all). The full [encoder-decoder architecture from Phase 02 Lesson 4](../../Phase-02-Transformer-Architecture-Deep-Dive/04-Transformer-Encoder-Decoder/README.md) is the model family that gets both: bidirectional understanding of an input *and* autoregressive generation of an output, connected by cross-attention. T5 and BART are the two most influential ways this shape got trained, and each introduces a genuinely new pretraining objective beyond the plain causal LM and MLM you've already implemented.

## What this lesson covers

- T5's "text-to-text" framing: every NLP task as one unified format
- T5's pretraining objective: span corruption
- BART: a bidirectional encoder + an autoregressive decoder, pretrained via denoising
- When encoder-decoder still beats decoder-only in practice
- Why decoder-only ended up dominating general-purpose LLMs anyway

## 1. T5's text-to-text framing

T5 (Raffel et al., 2020 — "Text-to-Text Transfer Transformer") makes a simple but influential observation: translation, summarization, classification, and regression can *all* be phrased as "take text in, produce text out," using a task prefix to tell the model which task to perform:

```
input:  "translate English to German: The house is wonderful."
output: "Das Haus ist wunderbar."

input:  "summarize: <long article text...>"
output: "<short summary text>"

input:  "cola sentence: The cat sat the mat."   (grammar acceptability judgment)
output: "unacceptable"
```

Every task uses the exact same model, the exact same loss (cross-entropy over generated tokens), and the exact same encoder-decoder architecture from [Phase 02 Lesson 4](../../Phase-02-Transformer-Architecture-Deep-Dive/04-Transformer-Encoder-Decoder/README.md) — only the text changes. This is the encoder-decoder world's version of decoder-only's "phrase everything as next-token prediction" unification from [Lesson 1 §5](../01-Decoder-Only-Models-GPT-Family/README.md#5-why-decoder-only-won).

## 2. T5's pretraining objective: span corruption

Rather than masking individual tokens (BERT's MLM, [Lesson 2 §2](../02-Encoder-Only-Models-BERT-Family/README.md#2-masked-language-modeling-mlm)), T5 corrupts **contiguous spans** of the input, replacing each entire span with a single sentinel token, and trains the decoder to reconstruct exactly the missing spans (and nothing else) as its target sequence:

```
original:  "the quick brown fox jumps over the lazy dog"
input:     "the quick <X> jumps over the <Y> dog"
target:    "<X> brown fox <Y> lazy <Z>"
```

Notice the target is *much shorter* than the input — it only contains the missing pieces, tagged by which sentinel they fill in for. This makes T5's pretraining computationally cheaper per example than reconstructing the entire sequence, while still requiring the encoder to understand the whole (corrupted) input and the decoder to generate fluent, correctly-ordered replacements.

## 3. BART: bidirectional encoder + autoregressive decoder, denoising pretraining

BART (Lewis et al., 2019) uses the identical architecture shape but a different training philosophy: **corrupt the input text in one of several ways, then train the model to reconstruct the *entire original, uncorrupted* text** as the decoder's target (unlike T5, which only reconstructs the missing spans). BART's paper experiments with several corruption strategies:

- **Token masking** — replace random tokens with `[MASK]` (BERT-style)
- **Token deletion** — remove random tokens entirely (the model must also figure out *where* something is missing)
- **Sentence permutation** — shuffle the order of sentences within the document
- **Document rotation** — start the document at a random point, model must identify the true beginning

Because the decoder is a full autoregressive Transformer (not just a classification head over masked positions like BERT's), BART can be trained to reconstruct *fluent full text*, which is exactly why it excels particularly at generation-heavy tasks like summarization and text infilling, while retaining BERT-like bidirectional understanding on the encoder side.

## 4. When encoder-decoder still wins

Despite decoder-only's dominance for general-purpose assistants, encoder-decoder models remain a strong, sometimes better, choice specifically when:

- the task has a **clean, structural separation** between input and output (translation, summarization) rather than open-ended continuation
- the **output length differs dramatically** from the input length in a task-consistent way
- you want the *input* to get genuinely bidirectional processing (every input token sees the whole input) without paying the cost of bidirectional processing for the *growing output* as well (the decoder still only needs causal self-attention over what it has generated so far, plus cross-attention back to the already-fully-encoded input)

## 5. Why decoder-only won anyway, for general-purpose LLMs

Maintaining two separate stacks, two attention patterns, and a training/inference pipeline that has to decide "what's the input, what's the output" adds real engineering complexity that doesn't pay for itself once a single decoder-only model, prompted well enough (see [Phase 07](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/README.md)), can handle translation, summarization, *and* open-ended chat *and* code *and* reasoning inside one unified interface. T5 and BART's ideas didn't disappear, though — span corruption and denoising pretraining directly influenced how later models think about pretraining objectives, and encoder-decoder models are still the default choice for dedicated, high-volume translation and summarization systems today.

## Video Script Outline

1. Motivation — "the full architecture from Phase 02, and the two most famous ways to train it"
2. T5: text-to-text framing with a task prefix, live example
3. T5's span-corruption objective, worked by hand
4. BART: denoising pretraining, the four corruption strategies
5. When encoder-decoder genuinely wins over decoder-only, concretely
6. Walkthrough of `example.py` — build T5-style span-corruption pairs and BART-style noising functions from scratch
7. Recap: three architecture families now covered -> preview Lesson 4 (Mixture of Experts) as an orthogonal axis of variation

## Further Reading

- Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)
- Lewis et al. (2019), *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*
- Devlin et al. (2018), *BERT* — useful to re-read after this lesson, to see exactly what T5/BART generalized beyond masked single tokens

# Pretraining Objectives

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `02-Pretraining-Objectives`

## Why this matters

[Lesson 1](../01-Pretraining-Data-Pipeline/README.md) got you a large pile of clean text. This lesson answers the next question: *what exact training signal do you extract from it?* This course has already built three different answers across Phases 02-03 — causal language modeling ([Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction)), masked language modeling ([Phase 03 Lesson 2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#2-masked-language-modeling-mlm)), and span corruption ([Phase 03 Lesson 3](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#2-t5s-pretraining-objective-span-corruption)) — but scattered across different architecture lessons. This lesson puts all three (plus a fourth, prefix LM) **side by side on the same sentence**, so the real insight becomes unmissable: these are not different architectures, they are different **loss-masking recipes** you can apply to fundamentally the same Transformer stack.

## What this lesson covers

- Recap: causal LM, masked LM, and span corruption, now compared directly
- Prefix LM: a fourth hybrid variant (bidirectional prefix, causal continuation)
- Exactly which token positions contribute to the loss under each objective, and what target each position predicts
- Which objective suits which architecture family, and why
- The "one architecture, many recipes" takeaway

## 1. The four objectives, at a glance

| Objective | Attention pattern | What gets predicted | Loss computed over |
|---|---|---|---|
| **Causal LM** ([Phase 02 L6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)) | Causal (each position sees only the past) | Every position predicts the *next* token | Every position in the sequence |
| **Masked LM** ([Phase 03 L2](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md)) | Full bidirectional (no mask) | ~15% of positions predict their *own original* token, from context | Only the masked positions |
| **Span corruption** ([Phase 03 L3](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md)) | Bidirectional encoder + causal decoder | Decoder predicts the *missing spans*, concatenated after sentinels | Every position in the (short) target sequence |
| **Prefix LM** (this lesson) | Bidirectional over a prefix, causal over the rest | Only the non-prefix positions predict the *next* token | Only the non-prefix positions |

## 2. Recap: causal, masked, and span corruption

**Causal LM** feeds tokens `0..T-2` and supervises every position to predict the very next token, using a strict causal mask so no position can "cheat" by looking ahead — exactly the recipe trained in [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction). It's the natural fit for **decoder-only** models because generation and training use the *identical* forward-pass shape: at inference, you also only ever have "everything so far" and need to predict what comes next.

**Masked LM** can't use a causal mask at all — the whole point of an **encoder-only** model is that every position sees the entire sequence, both directions, in one pass ([Phase 03 Lesson 2 §1](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#1-just-the-encoder-stack--bidirectional-by-construction)). Since a bidirectional position could otherwise trivially "predict" itself by copying its own input, MLM instead corrupts ~15% of *input* positions (mostly to `[MASK]`) and only asks the model to recover the *original* identity of those corrupted positions, using loss `-100`/ignore-index everywhere else.

**Span corruption** is what an **encoder-decoder** model does with the same underlying idea as MLM (corrupt part of the input, predict what was removed), but restructured around having a full autoregressive decoder available: instead of masking individual scattered tokens, it removes contiguous *spans*, and the decoder only has to generate the missing content (tagged by sentinel), not the whole original sequence — a much shorter, cheaper target ([Phase 03 Lesson 3 §2](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#2-t5s-pretraining-objective-span-corruption)).

## 3. Prefix LM: a fourth, hybrid variant

Prefix LM (used in UniLM, and in a variant of PaLM's training) asks: *what if you kept a single decoder-only stack, but relaxed the causal mask over just the first part of the sequence?* Concretely, split each training sequence into a **prefix** (say, the first `k` tokens) and a **continuation**:

```
positions 0..k-1  (prefix):        every position can attend to every OTHER prefix position, bidirectionally
positions k..T-1  (continuation):  each position attends causally -- to the whole prefix, plus everything
                                    in the continuation up to and including itself
```

Loss is computed **only over the continuation positions**, each predicting its next token, exactly like causal LM — the prefix positions contribute no loss at all, they exist purely to provide richer, bidirectionally-processed context. This gets you a genuinely useful property without needing two separate stacks: bidirectional understanding of a "question" or "instruction" portion of the input, combined with ordinary autoregressive generation for the "answer" portion — all inside one shared set of weights and one attention implementation, just with a different mask shape than either pure causal LM or pure MLM. Mechanically, this is just the causal mask from Lesson 1 with its upper-left `k x k` block un-masked.

## 4. Which objective suits which architecture, and why

The pattern is not a coincidence — each objective is shaped by what the corresponding architecture needs to be good at:

- **Decoder-only + causal LM**: training and inference have the *same* computational shape (predict next token from everything so far), which is exactly why decoder-only models are the natural choice for open-ended generation — see [Phase 03 Lesson 1 §5](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#5-why-decoder-only-won).
- **Encoder-only + MLM**: the architecture has no notion of "generate the next token" at all (every position already sees the whole input), so the only way to get a self-supervised signal is to hide *some* information and ask for it back — bidirectional understanding tasks (classification, embeddings) are the payoff.
- **Encoder-decoder + span corruption / denoising**: the encoder gets to bidirectionally process a corrupted input exactly like MLM's philosophy, but the decoder is a full generative stack, so the target can be free-form generated text rather than a fixed-vocabulary classification over masked slots — this is what makes T5/BART good at input-to-output generation tasks like summarization and translation.
- **Prefix LM**: sits deliberately in between — a single decoder-only-shaped stack that borrows bidirectional context for a "conditioning" portion, useful for instruction-like inputs, without the engineering cost of a second, separate decoder stack ([Phase 03 Lesson 3 §5](../../Phase-03-LLM-Architectures-and-Types/03-Encoder-Decoder-Models-T5-BART/README.md#5-why-decoder-only-won-anyway-for-general-purpose-llms)).

## 5. The takeaway: same Transformer, different recipe

`example.py` takes one toy tokenized sentence and applies all four masking recipes to it, printing exactly which input each objective feeds the model, which target it asks the model to predict at each position, and which positions actually contribute to the loss. The underlying self-attention and feed-forward math ([Phase 02](../../Phase-02-Transformer-Architecture-Deep-Dive/README.md) in full) never changes across the four — only the attention mask shape and the loss mask change. Internalizing this is the real point of the lesson: "decoder-only vs. encoder-only vs. encoder-decoder" is a difference in *masking recipe*, layered on top of one shared architectural toolkit.

## Video Script Outline

1. Motivation — "three objectives already met across two phases, now side by side for the first time"
2. Quick recap table: causal LM, MLM, span corruption
3. Introducing prefix LM as the fourth variant — the "bidirectional prefix, causal continuation" mask
4. Why each objective matches its architecture family's actual use case
5. Walkthrough of `example.py` — one sentence, four masking recipes, one printed comparison
6. The big takeaway: one shared Transformer toolkit, four different training recipes
7. Recap + preview of Lesson 3: once the objective is fixed, how do you actually train at scale across many machines?

## Further Reading

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1, causal LM)
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (MLM)
- Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, span corruption)
- Dong et al. (2019), *Unified Language Model Pre-training for Natural Language Understanding and Generation* (UniLM — the prefix LM masking idea)
- Chowdhery et al. (2022), *PaLM: Scaling Language Modeling with Pathways* (trains a variant with a prefix LM objective at scale)

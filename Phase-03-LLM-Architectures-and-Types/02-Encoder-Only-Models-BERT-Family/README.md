# Encoder-Only Models: the BERT Family

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `02-Encoder-Only-Models-BERT-Family`

## Why this matters

Lesson 1 covered why decoder-only won for *general-purpose, generative* LLMs. But encoder-only models solve a genuinely different problem better, and understanding exactly why makes the architectural trade-off concrete instead of just asserted. BERT (Devlin et al., 2018) also introduced **masked language modeling**, a pretraining objective fundamentally different from the causal next-token prediction used everywhere else in this course — training your own tiny version of it in `example.py` will make the encoder-only/decoder-only distinction viscerally clear, not just conceptual.

## Architecture at a glance

```
   [CLS] tok1 tok2 ... [SEP]
            │
   token embedding + positional embedding
            │
   ┌────────▼─────────────────────┐
   │      Encoder Block × N        │
   │  ┌──────────────────────┐     │
   │  │Bidirectional Self-Attn│     │  EVERY position attends to EVERY
   │  └──────────┬───────────┘     │  other position — before AND after,
   │        + residual             │  in one pass (no causal mask at all)
   │  ┌──────────▼───────────┐     │
   │  │     Feed-Forward      │     │
   │  └──────────┬───────────┘     │
   │        + residual             │
   └─────────────┼─────────────────┘
           final LayerNorm
                 │
     ┌───────────┴────────────┐
     ▼                         ▼
[CLS] vector              per-token vectors
     │                         │
sentence-level head      token-level head
(classification,          (NER, extractive QA)
 entailment, ...)

     no output head produces NEXT tokens — nothing here is autoregressive,
     so the model structurally cannot generate open-ended text
```

Same block shape as the decoder from [Lesson 1](../01-Decoder-Only-Models-GPT-Family/README.md#architecture-at-a-glance) — the entire architectural difference is *no causal mask*. `example.py` builds and trains exactly this stack from scratch with real MLM masking.

## What this lesson covers

- The bidirectional encoder stack, and why it can't generate open-ended text
- Masked Language Modeling (MLM): BERT's core pretraining objective
- Next Sentence Prediction (NSP), and why it was later dropped
- `[CLS]` and `[SEP]`: how BERT packages input for classification tasks
- Fine-tuning BERT for downstream tasks
- RoBERTa and the encoder-only family today

## 1. Just the encoder stack — bidirectional by construction

Recall [Phase 02 Lesson 4 §1](../../Phase-02-Transformer-Architecture-Deep-Dive/04-Transformer-Encoder-Decoder/README.md#1-the-encoder-stack): the encoder's self-attention has **no causal mask** — every position can attend to every other position, both before and after it, in the same forward pass. BERT is literally just this stack, with no decoder at all. This bidirectional context is exactly what makes encoder-only models excel at *understanding* tasks (classification, named-entity recognition, extractive question answering) where the whole input is available upfront — and exactly what makes them structurally unable to do open-ended generation: there is no notion of "generate the next token" when every position can already see every other position, including ones that haven't been "generated" yet.

## 2. Masked Language Modeling (MLM)

You can't train a bidirectional model with plain next-token prediction — a position that can see the future would trivially "predict" it by just copying it. BERT's fix: randomly select ~15% of input tokens, and train the model to predict *those specific tokens* from bidirectional context, having replaced the actual token with:

- the `[MASK]` token, 80% of the time
- a random other token, 10% of the time
- the original (unchanged) token, 10% of the time

The last two cases exist specifically so the model can't simply "learn to ignore non-`[MASK]` positions" — it has to build a genuinely useful representation at every position, since it never knows for certain which positions are being evaluated. Loss is computed **only over the masked positions** — this is the key mechanical difference from [Phase 01/02's causal language modeling loss](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction), which supervises every position in the sequence.

## 3. Next Sentence Prediction (NSP)

BERT's original pretraining also included a second objective: given two text segments, predict whether the second genuinely follows the first in the source document, or is a random unrelated segment — a binary classification task meant to teach sentence-pair relationships useful for tasks like question answering. Later work (notably RoBERTa, see below) found NSP contributed little or even hurt performance, and most modern encoder-only models drop it entirely — a useful reminder that not every idea in an influential paper survives follow-up scrutiny.

## 4. `[CLS]` and `[SEP]`: packaging input for classification

BERT prepends a special `[CLS]` token to every input, and uses `[SEP]` to separate segments (e.g., a question and a passage):

```
[CLS] the movie was great [SEP]
```

After pretraining, the final-layer representation *at the `[CLS]` position* is treated as a pooled, whole-sequence summary — a small classification head on top of just that one vector handles sentence-level tasks (sentiment, entailment), while token-level tasks (NER, extractive QA) use the per-token output vectors directly.

## 5. Fine-tuning BERT

Exactly the GPT-1 recipe from [Lesson 1 §1](../01-Decoder-Only-Models-GPT-Family/README.md#1-gpt-1-pretrain-then-fine-tune), applied to a bidirectional model: pretrain once with MLM (+ optionally NSP) on unlabeled text, then attach a small task-specific head and fine-tune the whole model (or just the head) on labeled data for a specific downstream task. This pretrain-then-fine-tune pattern predates and directly parallels GPT's — the two papers arrived at structurally similar recipes from opposite architectural starting points, within months of each other.

## 6. RoBERTa and today's encoder-only landscape

RoBERTa (Liu et al., 2019) re-ran BERT's recipe with more data, longer training, dynamically-generated masks (BERT precomputed masks once; RoBERTa remasks every epoch), and **no NSP objective**, and beat BERT across the board — evidence that BERT was undertrained relative to its capacity, not that its architecture was wrong. Encoder-only models mostly faded from headline "chatbot" news as decoder-only LLMs took over general-purpose use, but they remain the standard choice today for: producing text **embeddings** for search/similarity (their bidirectional pooled representations tend to be stronger than a decoder-only model's for this), classification pipelines, and any latency-sensitive task where you need strong understanding but no generation at all.

## Video Script Outline

1. Motivation — "the encoder half of Phase 02, alone — what does bidirectional-only buy you, and what does it cost?"
2. Why bidirectional self-attention structurally can't do open-ended generation
3. Masked Language Modeling: the 80/10/10 masking recipe, loss only on masked positions
4. NSP, and why it got dropped (RoBERTa)
5. `[CLS]`/`[SEP]` and the fine-tuning recipe
6. Walkthrough of `example.py` — train a tiny MLM model from scratch, contrast directly with Phase 02's causal LM training
7. Recap: encoder-only's niche today -> preview Lesson 3's middle ground (encoder-decoder)

## Further Reading

- Devlin, Chang, Lee, Toutanova (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Liu et al. (2019), *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* (a preview of the compression ideas in [Phase 09: Model Distillation and Pruning](../../Phase-09-Deployment-and-Inference-Optimization/05-Model-Distillation-and-Pruning/README.md))

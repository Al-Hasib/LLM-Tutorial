# Decoder-Only Models: the GPT Family

**Phase:** [LLM Architectures and Types](../README.md) · **Topic folder:** `01-Decoder-Only-Models-GPT-Family`

## Why this matters

You already built a working decoder-only Transformer in [Phase 02's capstone](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md). This lesson zooms out: that exact architecture family — scaled up, trained longer, on more data — *is* GPT-1, GPT-2, GPT-3, and (with a handful of refinements covered later in this phase) nearly every general-purpose LLM in production today, including the open models surveyed in [Lesson 7](../07-Survey-of-Popular-Open-LLMs/README.md). Understanding why decoder-only won out over the encoder-only and encoder-decoder alternatives (covered in Lessons 2 and 3) is essential context for the rest of this course.

## Architecture at a glance

```
        token ids
            │
   token embedding + positional embedding
            │
   ┌────────▼─────────────────────┐
   │      Decoder Block × N        │
   │  ┌──────────────────────┐     │
   │  │ Causal Self-Attention │     │  each position may only attend to
   │  └──────────┬───────────┘     │  ITSELF and positions BEFORE it
   │        + residual             │  (upper-triangular mask)
   │  ┌──────────▼───────────┐     │
   │  │     Feed-Forward      │     │
   │  └──────────┬───────────┘     │
   │        + residual             │
   └─────────────┼─────────────────┘
           final LayerNorm
                 │
        Linear head → vocab logits
                 │
     softmax → sample next token → feed back in
     (autoregressive: repeat one token at a time)
```

No encoder, no cross-attention — just the block above stacked `N` times. Every task becomes "predict the next token," which is exactly why the same architecture handles pretraining, fine-tuning, and open-ended generation with zero structural changes. `example.py` builds this stack as real, trainable PyTorch code (not just the parameter-count formulas below) and generates text from it before and after training.

## What this lesson covers

- GPT-1: decoder-only pretraining + task-specific fine-tuning
- GPT-2: scaling up, byte-level BPE, and zero-shot task transfer
- GPT-3: in-context learning emerges from scale alone
- How GPT's architecture evolved from Phase 02's mini-GPT to real production scale
- Why decoder-only became the default choice for general-purpose LLMs

## 1. GPT-1: pretrain, then fine-tune

Radford et al. (2018) introduced the recipe: pretrain a decoder-only Transformer with plain next-token prediction ([Phase 01](../../Phase-01-Language-Modeling-Foundations/01-What-is-a-Language-Model/README.md)) on unlabeled text, then **fine-tune** the same model with a small added output head for each specific downstream task (sentiment classification, entailment, etc.). This "pretrain once, fine-tune per task" pattern — as opposed to training a fresh model per task from scratch — was the paper's central contribution, and it's the direct ancestor of [Phase 05: Fine-tuning LLMs](../../Phase-05-Finetuning-LLMs/README.md).

## 2. GPT-2: scale, byte-level BPE, and zero-shot transfer

GPT-2 (Radford et al., 2019) kept the same architecture family but scaled it up substantially (four sizes, 117M to 1.5B parameters), switched to **byte-level BPE** tokenization (recall [Phase 02 Lesson 1 §5](../../Phase-02-Transformer-Architecture-Deep-Dive/01-Tokenization/README.md#5-byte-level-bpe-what-gpt-family-models-actually-use) — this is literally where that idea came from), and trained on a much larger, more diverse web-scraped corpus (WebText). The headline result: with enough scale, the model started performing reasonably on tasks it was **never explicitly fine-tuned for** — summarization, translation, question answering — just by being prompted correctly. This is **zero-shot task transfer**, and it was the first strong hint that scale itself, not task-specific engineering, was the more promising direction.

## 3. GPT-3: in-context learning emerges

GPT-3 (Brown et al., 2020) pushed scale further still (175B parameters) and made the zero-shot observation from GPT-2 far more dramatic and reliable: the model could perform new tasks from just a handful of examples placed directly in the prompt, **with no gradient updates at all** — this is **few-shot in-context learning**, and it's the capability [Phase 07: Prompt Engineering and In-Context Learning](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/README.md) is built entirely around. Nothing about the architecture changed to enable this — it's the same decoder-only Transformer and the same next-token-prediction objective as GPT-1, just at roughly 1500x the parameter count.

## 4. Architecture growth, concretely

| Model | Layers | `d_model` | Heads | Context length | Parameters |
|---|---|---|---|---|---|
| GPT-1 | 12 | 768 | 12 | 512 | ~117M |
| GPT-2 (small → XL) | 12 → 48 | 768 → 1600 | 12 → 25 | 1024 | 117M → 1.5B |
| GPT-3 | 96 | 12288 | 96 | 2048 | 175B |

Every row is the *exact same block* from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#2-the-full-model) — causal self-attention + feed-forward, stacked `N` times — just with bigger numbers. `example.py` computes real parameter counts from these configurations using the same formula scaling-laws research relies on (previewed here, covered fully in [Lesson 5](../05-Scaling-Laws/README.md)).

## 5. Why decoder-only won

Compare the three architecture families side by side (Lessons 1-3 of this phase): encoder-only models ([Lesson 2](../02-Encoder-Only-Models-BERT-Family/README.md)) can't generate open-ended text at all; encoder-decoder models ([Lesson 3](../03-Encoder-Decoder-Models-T5-BART/README.md)) need a clean separation between "input" and "output" that most real tasks (open-ended chat, reasoning, code) don't naturally have. Decoder-only models sidestep both problems: **any task can be phrased as "continue this text"** — question answering, summarization, translation, classification, chatting, all become the exact same next-token-prediction objective with different prompts. One architecture, one training objective, one tokenizer, one model — a much simpler system to scale, and scale turned out to matter more than architectural cleverness.

## Video Script Outline

1. Motivation — "the model you already built, just bigger — three times, with three different lessons learned"
2. GPT-1: pretrain + fine-tune paradigm
3. GPT-2: scale + byte-level BPE -> zero-shot transfer emerges
4. GPT-3: more scale -> few-shot in-context learning, no gradient updates
5. Architecture table walkthrough, tie every row back to Phase 02's mini-GPT block
6. Walkthrough of `example.py` — compute real GPT-2-family parameter counts from configs, then build and train the actual decoder-only block from scratch, generating text before and after training
7. Recap: why "one architecture, one objective, phrase everything as text continuation" won -> preview Lessons 2-3 for what got left behind

## Further Reading

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1)
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3)
- Wei et al. (2022), *Emergent Abilities of Large Language Models* (the broader phenomenon of capabilities appearing with scale)
- Sanh et al. (2019), *DistilBERT* — the distillation recipe it introduces applies just as well to decoder-only models (e.g. DistilGPT2); see [Phase 09 Lesson 4: Model Distillation and Pruning](../../Phase-09-Deployment-and-Inference-Optimization/04-Model-Distillation-and-Pruning/README.md) for the full mechanism

# Pretraining a Small LLM From Scratch

**Phase:** [Pretraining LLMs](../README.md) · **Topic folder:** `05-Pretraining-a-Small-LLM-From-Scratch`

## Why this matters

This lesson is the capstone of the pretraining phase — and it is *not* a repeat of [Phase 02 Lesson 6's mini-GPT](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md). That earlier lesson proved the *architecture* trains at all. This lesson proves the *recipe* — every piece built across this phase, combined into one run: a data pipeline step ([Lesson 1](../01-Pretraining-Data-Pipeline/README.md)), the causal-LM objective ([Lesson 2](../02-Pretraining-Objectives/README.md)), and AdamW with gradient clipping and a warmup+cosine schedule ([Lesson 4](../04-Mixed-Precision-and-Optimization/README.md)) — with the two additions every real pretraining run has and Phase 02's demo skipped: a genuinely held-out **validation set** (so you can tell memorization from generalization) and **checkpointed generation samples** across training (so "the model is learning" is something you watch happen, not just a loss number going down).

## What this lesson covers

- Assembling a tiny data-pipeline step (quality filter + exact dedup) before training even starts
- Splitting held-out validation data and tracking train vs. validation loss separately
- Gradient clipping: a new, previously unintroduced stability technique
- Wiring together AdamW + warmup/cosine schedule + gradient clipping into one training loop
- Watching generation quality improve at several checkpoints across training
- An honest scale comparison: exactly what's toy here, and what a real pretraining run adds

## 1. The pipeline step: filter and deduplicate before training

Recall [Lesson 1](../01-Pretraining-Data-Pipeline/README.md#3-quality-filtering) — bad data doesn't just waste compute, it actively teaches bad habits, and duplicate data wastes capacity re-memorizing the same string. This lesson's raw corpus is deliberately built with a few injected problems (some junk/low-quality documents, some exact duplicates) so `example.py` §1 can run a real (if simplified) quality filter and exact-deduplication pass, and show the resulting corpus is smaller and cleaner than the raw input, before a single training step ever runs.

## 2. Train/validation split: telling memorization from generalization

Every earlier from-scratch training demo in this course (Phase 02 Lesson 6's mini-GPT included) trained and generated from the *same* corpus, which makes it impossible to tell whether falling loss means the model is learning general structure or just memorizing the specific training text. This lesson holds out a **validation set** — text the model never trains on — and tracks loss on both throughout training. A widening gap between shrinking training loss and flat (or rising) validation loss is exactly what **overfitting** looks like in the metric itself, not just as a word — and `example.py` reports both numbers side by side.

## 3. Gradient clipping: a new stability technique

One piece from a real pretraining recipe not yet introduced: **gradient clipping**. Occasionally, a single batch produces an unusually large gradient (a token sequence the model is currently very wrong about) — left alone, this can trigger a destructively large optimizer step that knocks training off course, especially early on during [Lesson 4's warmup phase](../04-Mixed-Precision-and-Optimization/README.md#4-learning-rate-schedules-warmup-then-decay) when the model is least stable. Gradient clipping rescales the *entire* gradient vector (across all parameters at once) if its overall norm exceeds a threshold `max_norm`, preserving its direction but capping its magnitude:

```
if ||g|| > max_norm:
    g = g * (max_norm / ||g||)
```

This is a one-line addition to any training loop (`torch.nn.utils.clip_grad_norm_` in PyTorch) and is standard in essentially every real LLM pretraining run.

## 4. The full recipe, assembled

```
raw corpus -> quality filter + dedup (Lesson 1)
           -> train/validation split
           -> tokenize
           -> for each step:
                sample a batch (causal LM objective, Lesson 2)
                forward pass -> loss
                backward pass -> gradients
                clip gradient norm (Section 3, above)
                AdamW step at the current warmup/cosine learning rate (Lesson 4)
                periodically: log train AND validation loss, generate a sample
```

Every line of this traces back to an earlier lesson in this phase except gradient clipping, introduced here for the first time because a full training run is the first place in this course long/unstable enough to need it.

## 5. What's toy here, and what a real run adds

`example.py` trains for about a minute or two on a CPU, on a hand-written corpus of a few thousand characters, with a model of a few hundred thousand parameters. A real pretraining run differs by scaling *every one* of these numbers by many orders of magnitude — [distributed training](../03-Distributed-Training-Basics/README.md) across thousands of accelerators, [mixed-precision](../04-Mixed-Precision-and-Optimization/README.md#1-fp32-vs-fp16-vs-bf16) arithmetic, a [data pipeline](../01-Pretraining-Data-Pipeline/README.md) processing terabytes instead of kilobytes, and a token budget chosen according to [Phase 03's scaling laws](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) rather than "whatever finishes in two minutes." What does **not** change: the causal-LM loss, the AdamW+warmup+cosine+clipping recipe, and the train/validation-split discipline in this exact lesson are the same recipe used at every scale, all the way up to the largest models in this course's survey ([Phase 03 Lesson 7](../../Phase-03-LLM-Architectures-and-Types/07-Survey-of-Popular-Open-LLMs/README.md)).

With this lesson, the pretraining phase is complete — [Phase 05](../../Phase-05-Finetuning-LLMs/README.md) picks up exactly where a pretrained base model like this one leaves off: adapting it to follow instructions instead of just continuing text.

## Video Script Outline

1. Motivation — "not a repeat of Phase 02's mini-GPT: the full recipe, not just the architecture"
2. Walk through the raw corpus's injected problems, then the filter+dedup pass live
3. Train/validation split, and why watching both loss curves matters
4. Gradient clipping: the one new ingredient, and the intuition for why it's needed
5. Walkthrough of `example.py` — the assembled training loop, warmup/cosine LR schedule ticking as it trains
6. Watch generation samples improve across checkpoints, side by side
7. Read the final train-vs-validation loss gap together
8. Recap the entire pretraining phase's arc, and hand off to Phase 05's fine-tuning

## Further Reading

- Pascanu, Mikolov, Bengio (2013), *On the difficulty of training Recurrent Neural Networks* (the original gradient-clipping paper)
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3's Appendix B/C document a real version of exactly this recipe at scale)
- Karpathy, *nanoGPT* (github.com/karpathy/nanoGPT) — a real, runnable, larger-scale version of precisely the training loop built in `example.py`

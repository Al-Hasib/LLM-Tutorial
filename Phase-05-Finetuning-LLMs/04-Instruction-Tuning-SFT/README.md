# Instruction Tuning (SFT)

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `04-Instruction-Tuning-SFT`

## Why this matters

[Phase 03 Lesson 1 §1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#1-gpt-1-pretrain-then-fine-tune) already introduced the shape of this idea: GPT-1 pretrained on raw text, then fine-tuned on labeled data for a specific downstream task. **Instruction tuning** (often called **Supervised Fine-Tuning**, or **SFT**) is that exact same pretrain-then-fine-tune pattern, generalized: instead of fine-tuning toward one narrow task (sentiment classification, entailment), you fine-tune a base model on thousands of *(instruction, response)* pairs spanning many tasks at once, so it learns a general behavior — **follow whatever instruction you give it, and respond helpfully** — rather than just completing arbitrary text in whatever style happens to continue it statistically. This is the step that turns a raw pretrained language model into something that acts like an assistant, and it's the training stage that [Lesson 5&#39;s Hugging Face workflow](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) and [Lesson 6&#39;s case study](../06-Domain-Specific-Finetuning-Case-Study/README.md) both build directly on top of.

## What this lesson covers

- Why instruction tuning is the same architecture and objective as pretraining, with a different dataset
- Instruction data formats: role tags and chat templates
- The SFT loss: cross-entropy on response tokens only, masking the prompt with `ignore_index`
- How this differs from BERT's masked-language-modeling loss, despite both using a "mask"
- Why instruction *data quality and diversity* dominates raw quantity (the LIMA finding)
- What `example.py`'s from-scratch masked fine-tuning run actually demonstrates

## 1. Same model, same objective, different data

Nothing about the Transformer architecture or the next-token-prediction loss from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction) changes for instruction tuning. A pretrained base model — trained the way [Phase 04](../../Phase-04-Pretraining-LLMs/README.md) describes, on a huge corpus of raw, unlabeled text — already knows how to produce plausible continuations of *any* text. What it hasn't learned is a specific *behavior*: that when it sees something shaped like a question or a command, the "plausible continuation" it should produce is a direct, helpful answer, not just more text in whatever register the prompt happened to resemble (a forum post, a Wikipedia stub, a legal disclaimer — a raw base model has no bias toward "assistant" behavior at all). Instruction tuning fixes exactly that, by continuing ordinary supervised training on a curated dataset built entirely from *(instruction, response)* pairs, using [full fine-tuning or PEFT](../01-Full-Finetuning-vs-PEFT/README.md) — either is compatible with SFT; the *loss* and *data*, not the update mechanism, are what define instruction tuning.

## 2. Instruction data formats: role tags and chat templates

The simplest instruction format is just two labeled fields, popularized by early open instruction datasets like Stanford Alpaca:

```
### Instruction:
Summarize the following paragraph in one sentence.

### Response:
<the desired summary>
```

Production chat models use a richer version of the same idea — a **chat template** with explicit **role tags** marking who "said" each turn:

```
<|system|>
You are a helpful assistant.
<|user|>
Summarize the following paragraph in one sentence.
<|assistant|>
<the desired summary>
```

The special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, or model-specific equivalents) are added to the tokenizer's vocabulary and let a single model handle multi-turn conversations, system prompts, and tool outputs, all inside the same token stream the Transformer already processes uniformly. Hugging Face tokenizers expose this as `tokenizer.apply_chat_template(...)`, used throughout [Lesson 5](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md#2-data-instruction-formatting-and-chat-templates). `example.py` uses the simplest possible two-field version (`Instruction: ... \nResponse: ...`) so the masking mechanic below stays easy to see character-by-character.

## 3. The loss: cross-entropy on response tokens only

If you trained on the full `(instruction + response)` text with ordinary next-token-prediction loss over every position, the model would spend a large fraction of its gradient updates learning to *reproduce instructions* — predicting the next word of "Summarize the following paragraph..." — which is not the behavior you want and dilutes the signal that actually teaches it to respond well. The fix, universal across real SFT implementations: build a **label tensor** identical in shape to the input, but with the prompt portion replaced by an **ignore value** (PyTorch's convention is `-100`, `nn.CrossEntropyLoss`'s default `ignore_index`) so those positions contribute zero loss and zero gradient:

```
input:   Instruction : uppercase the word cat \n Response :   C   A   T  \n
labels:  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100  C   A   T  \n
```

Concretely, for a tokenized sequence, position `i`'s label supervises predicting token `i+1`; it is set to `-100` whenever token `i+1` still belongs to the prompt, and to the real token id otherwise. The one subtlety: the position right at the prompt/response boundary — predicting the *first* response token from the last prompt token — **is** supervised, since "given the full prompt, produce the first token of a response" is precisely the behavior being taught. `example.py`'s `build_sft_example` implements this exactly, and `show_masking_demo` prints every position's label so the mechanism is visible, not just asserted.

## 4. Not the same "mask" as BERT

It's easy to conflate this with [Phase 03 Lesson 2&#39;s Masked Language Modeling](../../Phase-03-LLM-Architectures-and-Types/02-Encoder-Only-Models-BERT-Family/README.md#2-masked-language-modeling-mlm), since both use a masking mechanism and both restrict the loss to a subset of positions — but they mask completely different things:

|                       | BERT's MLM                                                                                                                       | SFT                                                                                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What gets masked      | **Input** tokens are replaced with `[MASK]` (or noise)                                                                   | Nothing in the input changes at all                                                                                                                                                                  |
| What the loss ignores | Everything**except** the masked positions                                                                                  | Everything**inside the prompt**                                                                                                                                                                |
| Why                   | Force the model to build genuinely useful bidirectional representations, since it doesn't know which positions will be evaluated | Avoid wasting gradient on re-deriving the prompt; supervise only the behavior being taught                                                                                                           |
| Attention direction   | Bidirectional (encoder)                                                                                                          | Causal (decoder) —[Phase 02 Lesson 6 §1](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#1-the-model-token-positional-embedding-n-decoder-blocks-head) |

SFT's model still *sees* the full, unaltered prompt as input at every position (ordinary causal self-attention over real tokens); only the **loss target** is masked, and only over the prompt half of the sequence.

## 5. Data quality and diversity beat raw quantity

A natural instinct is "more instruction examples = a better assistant." Zhou et al. (2023), in the **LIMA** paper ("Less Is More for Alignment"), found this isn't the dominant factor: fine-tuning a strong pretrained base model on just **1,000 carefully curated, diverse, high-quality** instruction/response pairs produced outputs competitive with models trained on orders of magnitude more instruction data. Their explanation — the **Superficial Alignment Hypothesis** — is that a base model's knowledge and capabilities are learned almost entirely during large-scale *pretraining*; instruction tuning's job is comparatively small: teach the model the *style and format* of helpful responses, which a small, carefully diverse set of examples can already demonstrate thoroughly. This reframes the instruction-tuning-data question from "how much do I have" to "does it cover a wide enough range of tasks, phrasings, and response styles, at consistently high quality" — a data-curation lens [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md#1-data-curation-for-a-narrow-domain) applies directly to a narrow domain.

## 6. What `example.py` does

`example.py` builds the exact mechanism above from scratch:

1. Pretrains a tiny decoder-only Transformer (the same `MiniGPT` block from [Lesson 2](../02-LoRA-and-QLoRA/README.md) and [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md)) on generic text with plain, unmasked next-token prediction — standing in for "a base model that already knows the language."
2. Prints the masked label tensor for one instruction/response example, position by position, so `-100` vs. real targets is visible directly.
3. Shows the pretrained (not-yet-instruction-tuned) model's generation on a held-out instruction prompt — it rambles in generic-corpus style and never produces anything resembling a response.
4. Continues training the *same* model on a small toy instruction dataset (two task types — "uppercase" and "reverse" a word — applied to a set of training words) using the response-only masked loss, then re-runs generation on the same held-out prompt and measures exact-match accuracy on both the training words and a set of words *never seen during instruction tuning*.

The held-out numbers are reported honestly, whatever they turn out to be: with a model and dataset this tiny, instruction tuning reliably teaches the **format** (stop after a short answer, respond instead of rambling) even on unseen inputs, while the specific **task skill** generalizing perfectly to brand-new words needs more/more-diverse examples than this toy run provides — a small, concrete echo of §5's point that what and how much instruction data you use directly determines what actually generalizes.

## Video Script Outline

1. Motivation — "the exact fine-tuning step that turns a raw base model into something that acts like an assistant"
2. Recap: GPT-1's pretrain-then-fine-tune, generalized from one task to many instruction/response pairs
3. Instruction data formats: Alpaca-style fields, then real chat templates with role tags
4. The SFT loss: masking the prompt with `ignore_index=-100`, walked through position by position
5. Contrast with BERT's MLM masking — same word "mask," different mechanism and purpose
6. LIMA: why 1,000 good examples can beat a much larger, lower-quality dataset
7. Walkthrough of `example.py` — pretrain, show the label mask explicitly, instruction-tune, and compare before/after generation and accuracy on held-out words
8. Recap + preview: Lesson 5 does this exact same training with real Hugging Face tooling (`SFTTrainer`) at real model scale

## Further Reading

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1's original pretrain-then-fine-tune recipe)
- Wei et al. (2021), *Finetuned Language Models Are Zero-Shot Learners* (FLAN — instruction tuning across many tasks at once)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — SFT as the first stage before RLHF, previewed in [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md))
- Taori et al. (2023), *Stanford Alpaca: An Instruction-Following LLaMA Model* (the simple `### Instruction:` / `### Response:` format used as this lesson's toy format)
- Zhou et al. (2023), *LIMA: Less Is More for Alignment* (the data-quality-over-quantity finding in §5)

# Full Fine-tuning vs Parameter-Efficient Fine-tuning

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `01-Full-Finetuning-vs-PEFT`

## Why this matters

[Phase 04](../../Phase-04-Pretraining-LLMs/README.md) covered how a base model learns general language ability from raw text, and [Phase 03 Lesson 1 §1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#1-gpt-1-pretrain-then-fine-tune) already previewed the next step from GPT-1's original paper: pretrain once, then fine-tune for a specific behavior. This lesson is about the practical question that dominates real-world fine-tuning decisions: fine-tune *how*? Every parameter, or a small, cheap add-on? The answer determines whether fine-tuning a large model needs a roomful of GPUs or a single consumer card — and it sets up every other lesson in this phase, since [LoRA](../02-LoRA-and-QLoRA/README.md), [prompt/prefix tuning and adapters](../03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md), and the [Hugging Face PEFT/TRL workflow](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) are all specific answers to "how do we avoid full fine-tuning's cost?"

## What this lesson covers

- What full fine-tuning actually updates, and why it's expensive to train (not just to store)
- The AdamW optimizer-state memory tax, and where the "16 bytes per parameter" rule of thumb comes from
- Catastrophic forgetting: why updating every weight risks erasing general capability
- The general Parameter-Efficient Fine-Tuning (PEFT) idea: freeze the base, train a small add-on
- Multi-task deployment cost: one full model copy per task vs. one shared base + many tiny adapters

## 1. Full fine-tuning: update everything

Full fine-tuning takes a pretrained model's weights as the *initialization* and continues ordinary gradient descent — the exact same forward pass, cross-entropy loss, and backward pass from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md#3-training-objective-next-token-prediction) — except now every one of the model's existing parameters is a trainable leaf that receives gradient updates on new, task- or domain-specific data. Conceptually this is the simplest possible option, and it has the highest achievable quality ceiling: nothing about the model is restricted, so it can in principle reach any point in weight-space that plain pretraining could have reached. The cost is what makes it impractical past a certain scale, and that cost is not really about storing the weights — it's about *training* them.

## 2. The real memory cost: it's the optimizer, not the weights

Storing a model's weights alone is cheap relative to training it. Training with AdamW (the de facto standard optimizer for Transformers) requires, **per trainable parameter**:

```
weights (fp16)                     2 bytes
gradients (fp16)                   2 bytes
fp32 master copy of the weights    4 bytes   (kept for numerically stable updates)
Adam first moment  (m, fp32)       4 bytes
Adam second moment (v, fp32)       4 bytes
---------------------------------------------
total                             16 bytes / trainable parameter
```

This is the standard mixed-precision-training memory breakdown (see Rajbhandari et al., 2020, *ZeRO*, for the same accounting). The headline consequence: **full fine-tuning a 7-billion-parameter model needs roughly 16 x 7B ≈ 112 GB** just for weights + gradients + optimizer state — before a single activation is stored for the backward pass, and before any of the training batch's own memory. Merely *loading* that same model for inference, by contrast, needs only its 2-byte weights: about 14 GB. Full fine-tuning's memory demand is dominated entirely by the optimizer bookkeeping around parameters that get updated — which is exactly the quantity PEFT methods attack directly.

## 3. Catastrophic forgetting

Beyond cost, full fine-tuning carries a real quality risk: because every weight is free to move, gradient descent on a new, narrower dataset can overwrite the general knowledge and capabilities the base model spent enormous pretraining compute acquiring. This is **catastrophic forgetting** — the model gets better at the fine-tuning task while silently getting worse at everything else it used to be able to do. [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md) measures this trade-off directly and quantitatively. It's a second, independent reason (beyond raw memory cost) to prefer constraining *how much* of the model is allowed to change.

## 4. The PEFT idea: freeze almost everything

Parameter-Efficient Fine-Tuning (PEFT) is a family of techniques built around one shared strategy: **freeze the entire pretrained base model** (`requires_grad=False` on every original parameter — it contributes to the forward pass but never receives a gradient update) **and introduce a small number of new, trainable parameters** whose job is to steer or adjust the frozen model's behavior. Since only the small new piece needs gradients, a fp32 master copy, and Adam moments, the 16-bytes-per-parameter tax applies only to a sliver of the model:

```
Full fine-tuning memory  =  N_total x 16 bytes
PEFT memory              =  N_total x 2 bytes (frozen, inference-only)  +  N_trainable x 16 bytes
```

Since `N_trainable` is typically 0.01%-1% of `N_total` (concrete numbers for LoRA specifically are computed in [Lesson 2](../02-LoRA-and-QLoRA/README.md#3-counting-parameters)), the second term all but vanishes, and PEFT's total memory is dominated by the cheap frozen-inference cost of the base model — regardless of how large the base model is. `example.py` computes this comparison directly across a realistic range of model sizes.

## 5. The multi-task deployment angle

The memory argument matters most during *training*, but PEFT has a second, independent payoff at *deployment* time. If you need a model fine-tuned for five different tasks, full fine-tuning produces **five entire copies of the model's weights** (each one a full checkpoint, since every weight potentially changed). PEFT produces **one shared frozen base plus five tiny adapter files** — often single-digit megabytes each, versus gigabytes per full checkpoint — because only the small added parameters differ between tasks. This is why production systems serving many fine-tuned variants of the same base model (e.g., one LoRA adapter per customer or per task) almost always use PEFT: swapping adapters in and out of memory is vastly cheaper than swapping entire models.

## Video Script Outline

1. Motivation — "fine-tuning a 7B model: why does that need 100+ GB, when the model itself is 14 GB?"
2. Full fine-tuning recap: same training loop as pretraining, just continued from a checkpoint
3. Where the memory actually goes: weights, gradients, and the AdamW optimizer-state tax, in detail
4. Catastrophic forgetting as a second, independent cost of updating everything
5. The PEFT idea: freeze the base, train a tiny add-on, pay the 16-byte tax on almost nothing
6. Walkthrough of `example.py` — the memory-footprint calculator across model sizes, and the multi-task storage comparison
7. Recap + preview: Lesson 2 makes "the tiny add-on" concrete with LoRA

## Further Reading

- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP* (the paper that named and popularized the PEFT idea via adapters, covered fully in [Lesson 3](../03-Prompt-Tuning-Prefix-Tuning-Adapters/README.md))
- Rajbhandari, Rasley, Ruwase, He (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (the mixed-precision Adam memory accounting used above)
- Kirkpatrick et al. (2017), *Overcoming Catastrophic Forgetting in Neural Networks*

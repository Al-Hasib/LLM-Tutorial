# Fine-tuning with Hugging Face (PEFT + TRL)

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `05-Finetuning-with-HuggingFace-PEFT-TRL`

## Why this matters

Every mechanism this phase has built from scratch — [PEFT&#39;s frozen-base idea](../01-Full-Finetuning-vs-PEFT/README.md#4-the-peft-idea-freeze-almost-everything), [LoRA&#39;s `A`/`B` low-rank update](../02-LoRA-and-QLoRA/README.md#1-the-core-idea-freeze-w-learn-a-low-rank-update), and [SFT&#39;s response-masked cross-entropy loss](../04-Instruction-Tuning-SFT/README.md#3-the-loss-cross-entropy-on-response-tokens-only) — is exactly what a real fine-tuning job runs, just wrapped behind a small number of well-tested library calls instead of hand-written PyTorch. This lesson is the bridge from "I understand the mechanism" to "I can run this on a real pretrained model": Hugging Face's `transformers`, `peft`, and `trl` libraries, used together, are the closest thing the open-source LLM world has to a standard fine-tuning stack, and this is what you'd actually reach for outside a course built to teach internals. [Lesson 6](../06-Domain-Specific-Finetuning-Case-Study/README.md) then applies these same ideas — still from scratch, for full visibility into the results — to a concrete domain-adaptation case study.

**A note on this environment:** `transformers`, `peft`, and `trl` are not installed in this course's runtime (every other lesson deliberately sticks to raw PyTorch). `example.py` therefore checks for them, prints an install command and exits cleanly if they're missing — which is exactly what happens when you run it here — while every function in the file contains real, accurate library usage you can read directly, or run for real after `pip install transformers peft trl`.

## What this lesson covers

- `transformers.AutoModelForCausalLM` / `AutoTokenizer`: loading a real pretrained model and its tokenizer
- `peft.LoraConfig` and `get_peft_model`: applying LoRA to a real model, and what each config field controls
- Mapping every `LoraConfig` field back to Lesson 2's from-scratch `LoRALinear`
- `trl.SFTTrainer`: running the masked instruction-tuning loss from Lesson 4 via one `.train()` call
- Chat-template-based data formatting for `SFTTrainer`
- Merging and saving the fine-tuned adapter for deployment

## 1. Loading a real pretrained model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
```

`AutoModelForCausalLM` and `AutoTokenizer` are Hugging Face's "figure out which model/tokenizer class this checkpoint needs, and load it" entry points — the same two lines work whether the checkpoint name points to a 100M-parameter model or a 70B one. This *is* the base model every fine-tuning method in this phase assumes: pretrained via the recipe in [Phase 04](../../Phase-04-Pretraining-LLMs/README.md), downloaded here instead of trained from scratch.

## 2. `LoraConfig`, mapped directly to Lesson 2's `LoRALinear`

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Every field is a direct, named counterpart to something [Lesson 2](../02-LoRA-and-QLoRA/README.md) built by hand:

| `LoraConfig` field | What it controls                                                                                                                              | Lesson 2 equivalent                                                                                                                                                                                                                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `r`                | The LoRA rank                                                                                                                                 | `r` in `W' = W + (alpha/r) * B @ A` ([section 1](../02-LoRA-and-QLoRA/README.md#1-the-core-idea-freeze-w-learn-a-low-rank-update))                                                                                                                                                                                                                                        |
| `lora_alpha`       | The scaling numerator                                                                                                                         | `alpha` in the same formula — `lora_alpha / r` is the scaling factor `LoRALinear.scaling` computed directly                                                                                                                                                                                                                                                           |
| `lora_dropout`     | Dropout applied only to the LoRA path (`x @ A.T @ B.T`), not the frozen base                                                                | Not present in Lesson 2's minimal version — a regularizer real workflows add on top                                                                                                                                                                                                                                                                                         |
| `target_modules`   | *Which* frozen `nn.Linear` layers get wrapped with an `A`/`B` update                                                                  | Which weight matrix`LoRALinear` wraps — real configs typically target only the attention `q_proj`/`v_proj` (or `k_proj`/`o_proj`, or the FFN projections), not every linear layer, exactly because [Lesson 2 section 2](../02-LoRA-and-QLoRA/README.md#2-why-the-parameter-count-drops-so-much)'s parameter savings compound across however many matrices you skip |
| `bias`             | Whether bias terms are also trained (`"none"`, `"all"`, or `"lora_only"`)                                                               | Lesson 2's`base.bias` staying frozen (`bias="none"` is the common default)                                                                                                                                                                                                                                                                                               |
| `task_type`        | Tells`peft` which model "shape" it's wrapping (causal LM vs. sequence classification, etc.), so it knows where to attach adapters correctly | N/A — Lesson 2's`LoRALinear` only ever wrapped one `nn.Linear`, with no notion of "task"                                                                                                                                                                                                                                                                                |

`get_peft_model` walks the loaded model, replaces every `nn.Linear` whose name matches `target_modules` with a LoRA-wrapped version (functionally identical to `LoRALinear.forward` from Lesson 2), and freezes everything else automatically. `model.print_trainable_parameters()` prints the exact trainable-vs-frozen breakdown [Lesson 2 section 2](../02-LoRA-and-QLoRA/README.md#2-why-the-parameter-count-drops-so-much) computed by formula, now for a real loaded model.

## 3. Data: instruction formatting and chat templates

```python
def to_text(example):
    example["text"] = (
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    )
    return example

dataset = dataset.map(to_text)
```

A real workflow can use this simple Alpaca-style template, or a model's actual **chat template** via `tokenizer.apply_chat_template(messages, tokenize=False)`, which inserts the model-specific role tags described in [Lesson 4 section 2](../04-Instruction-Tuning-SFT/README.md#2-instruction-data-formats-role-tags-and-chat-templates) automatically, from a list of `{"role": ..., "content": ...}` dicts. Either way, `SFTTrainer` is told which dataset column holds the fully-formatted text via `dataset_text_field`.

## 4. `SFTTrainer`: Lesson 4's masked loss, in one call

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./sft-lora-demo",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    dataset_text_field="text",
    max_seq_length=128,
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

`SFTTrainer.train()` runs the exact loop [Lesson 4](../04-Instruction-Tuning-SFT/README.md) built by hand: tokenize each formatted example, build a label tensor with `-100` over the prompt portion (`SFTTrainer` detects the prompt/response split from the formatting or an explicit `response_template`), compute cross-entropy only on the response tokens, and step an optimizer — plus batching, padding, gradient accumulation, checkpointing, and logging handled for you. It is a thin, well-tested wrapper around `transformers.Trainer`, specialized for the instruction-tuning shape of data and loss.

## 5. Merging and deploying

```python
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./final-merged-model")
```

`merge_and_unload()` is `peft`'s real implementation of [Lesson 2 section 3](../02-LoRA-and-QLoRA/README.md#3-merging-at-inference-time-zero-extra-latency)'s `merged_weight()` method — it computes `W + (alpha/r) * B @ A` for every LoRA-wrapped layer, writes the result back into an ordinary `nn.Linear`, and returns a plain model with no `peft`-specific wrapper code left at inference time, at zero extra latency versus the original base model.

## 6. What if I don't have a GPU or these libraries installed?

Nothing in this lesson requires you to run it to understand it — every one of the code blocks above is real, working syntax as of recent `transformers`/`peft`/`trl` releases, and `example.py` contains them verbatim inside real functions. If the libraries genuinely aren't installed (as in this course's environment), running `example.py` prints an install command and exits cleanly rather than crashing. Installing them (`pip install transformers peft trl`) and re-running executes the actual small demo end to end, including a real LoRA adapter trained on a toy instruction set with `sshleifer/tiny-gpt2`, a tiny public checkpoint chosen specifically because it downloads and trains fast enough for a quick demo.

## Video Script Outline

1. Motivation — "everything built by hand in this phase, now as the library calls you'd actually use"
2. `AutoModelForCausalLM` / `AutoTokenizer`: loading a real pretrained checkpoint
3. `LoraConfig` field-by-field, mapped directly to Lesson 2's `LoRALinear`
4. Data formatting: Alpaca-style templates and real chat templates
5. `SFTTrainer.train()`: Lesson 4's masked loss, running as one call
6. `merge_and_unload()`: Lesson 2's merge-for-free property, for real
7. Walkthrough of `example.py`'s graceful ImportError handling, and why this course teaches the from-scratch version first
8. Recap + preview: Lesson 6 applies this whole stack to one concrete domain-adaptation case study

## Further Reading

- Hugging Face `transformers` documentation, `AutoModelForCausalLM` and `Trainer`
- Hugging Face `peft` documentation, `LoraConfig` and `get_peft_model`
- Hugging Face `trl` documentation, `SFTTrainer` and `SFTConfig`
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (the method `LoraConfig` implements — full derivation in [Lesson 2](../02-LoRA-and-QLoRA/README.md))
- von Werra et al. (2020), *TRL: Transformer Reinforcement Learning* (the library's origin — now covers SFT, DPO, and RLHF-style training; DPO revisited in [Phase 06 Lesson 4](../../Phase-06-Alignment-and-RLHF/04-Direct-Preference-Optimization-DPO/README.md))

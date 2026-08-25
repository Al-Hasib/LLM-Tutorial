# Prompt Tuning, Prefix Tuning and Adapters

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `03-Prompt-Tuning-Prefix-Tuning-Adapters`

## Why this matters

[Lesson 2](../02-LoRA-and-QLoRA/README.md) covered the PEFT method you'll actually reach for most often (LoRA), but it wasn't the first, and it isn't the only shape a "small trainable add-on to a frozen model" can take. This lesson covers three earlier, structurally different answers to the same question [Lesson 1](../01-Full-Finetuning-vs-PEFT/README.md) posed — freeze the base, train something small — each inserting its trainable piece in a different place in the model. Seeing all three side by side, plus LoRA, turns "PEFT" from one specific trick into a genuine design space with real trade-offs, which is exactly how the [Hugging Face `peft` library](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) presents it: LoRA, prompt tuning, prefix tuning, and adapters are all just different `PeftConfig` subclasses over the same frozen-base idea.

## What this lesson covers

- Prompt Tuning: trainable "soft prompt" embeddings prepended to the input
- Prefix Tuning: trainable key/value vectors prepended at every layer, not just the input
- Adapters: small trainable bottleneck MLPs inserted inside each Transformer layer
- Where each method's added parameters physically live, and why that placement matters
- A side-by-side trainable-parameter comparison against each other and against LoRA

## 1. Prompt Tuning: trainable tokens, frozen everything else

Lester, Al-Rfou, Constant (2021) proposed the simplest version of this idea: take `k` new, randomly-initialized embedding vectors (typically `k = 10`-`100`), **prepend** them to the input sequence's token embeddings, and freeze absolutely everything else in the model — including the ordinary token embedding table itself. Only these `k` new "soft prompt" vectors ever receive a gradient update:

```
input to layer 1 = [ soft_prompt_1, ..., soft_prompt_k, embed(token_1), ..., embed(token_T) ]
```

These vectors are not word embeddings of any real token — they're free-floating parameters that gradient descent shapes into whatever representation best steers the frozen model toward the desired task, functioning like a permanently-optimized, continuous (rather than discrete-word) prompt prepended ahead of the real input. Because they only influence the very first layer's input directly, their effect on deeper layers is indirect — every later layer only "feels" them through however much the first layer's self-attention propagated their influence forward.

## 2. Prefix Tuning: influence at every layer

Li and Liang (2021) proposed a closely related but more powerful variant, developed independently and around the same time: instead of prepending trainable vectors only to the *input* embeddings, prepend trainable vectors directly to the **keys and values of every Transformer layer's self-attention**:

```
layer i's attention keys   = [ prefix_K_i,  K_1, ..., K_T ]
layer i's attention values = [ prefix_V_i,  V_1, ..., V_T ]
```

Every layer gets its *own* trainable prefix vectors (a separate `prefix_K_i`, `prefix_V_i` pair per layer), so the added parameters can directly shape attention behavior deep inside the network, not just at the input. This gives prefix tuning noticeably more influence over the model's behavior than prompt tuning for a similar parameter budget, at the cost of more trainable parameters (scaling with the number of layers, not just once at the input) and a more invasive implementation (it needs access to every layer's attention computation, not just the embedding layer).

## 3. Adapters: small bottleneck MLPs inside every layer

Houlsby et al. (2019) — chronologically the earliest of the three, and the paper that first popularized the general "freeze the base, train a small add-on" PEFT strategy — took a different approach again: insert a small **bottleneck MLP module** after each sublayer (attention and feed-forward) inside every Transformer block:

```
Adapter(x) = x + W_up @ activation(W_down @ x)

W_down : bottleneck x d_model     (projects DOWN to a small bottleneck dimension)
W_up   : d_model x bottleneck     (projects back UP to the original width)
bottleneck << d_model                (e.g. 8-64, versus d_model in the hundreds/thousands)
```

The residual connection around the adapter (`x + ...`) means the adapter is initialized to output zero (via a near-zero-initialized `W_up`), so — exactly like LoRA's zero-initialized `B` matrix — training starts from the frozen model's original behavior and only gradually diverges. Unlike prompt/prefix tuning, adapters add a small amount of **extra compute to every forward pass** (two extra small matrix multiplications per adapter, per token) since they sit inline in the computation graph rather than just extending the input sequence or the attention keys/values.

## 4. Where the parameters live: a side-by-side comparison

| Method | Where trainable parameters live | Adds compute per forward pass? | Merges back to zero overhead? |
|---|---|---|---|
| Prompt Tuning | New embedding vectors, prepended once, at the input only | Slightly longer sequence | No |
| Prefix Tuning | New K/V vectors, prepended at every attention layer | Slightly longer effective sequence, every layer | No |
| Adapters | New bottleneck MLPs, inserted inside every layer | Yes — two extra small matmuls per layer | No |
| LoRA ([Lesson 2](../02-LoRA-and-QLoRA/README.md)) | Low-rank update alongside existing weight matrices | No extra compute during training; **zero** after merging | **Yes** — merges directly into existing weights |

LoRA's ability to merge back into the original weight matrices with literally zero inference-time overhead is a big part of why it became the default choice in practice — the other three methods all leave a small permanent tax on every inference forward pass (a longer effective sequence, or extra matmuls) that can't be merged away, since their trainable parameters don't share the same shape as any existing weight matrix.

## 5. Trainable-parameter counts, compared directly

For the same base model, the trainable parameter counts differ by orders of magnitude depending on the method and its hyperparameter (`k` soft-prompt tokens, adapter bottleneck size, or LoRA rank `r`). `example.py` computes and prints this comparison directly for a realistic model configuration, alongside two hands-on gradient-flow demonstrations: a working soft-prompt-tuning setup and a working adapter module, each verified to leave the frozen base model's gradients at exactly zero.

## Video Script Outline

1. Motivation — "LoRA wasn't the first idea here, and the alternatives put their trainable parameters in genuinely different places"
2. Prompt Tuning: soft embeddings prepended to the input, frozen everything else
3. Prefix Tuning: the same idea pushed into every layer's K/V, for more influence
4. Adapters: bottleneck MLPs inserted inline, with a zero-init residual
5. Side-by-side table: where parameters live, and the merge-to-zero-overhead property only LoRA has
6. Walkthrough of `example.py` — train a soft prompt and an adapter, verify frozen-gradient isolation, compare parameter counts across all four methods
7. Recap + preview: Lesson 4 uses full fine-tuning of a small model to focus purely on the instruction-tuning objective itself, before Lesson 5 shows the real Hugging Face `peft` API for all of these methods

## Further Reading

- Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP* (Adapters)
- Li and Liang (2021), *Prefix-Tuning: Optimizing Continuous Prompts for Generation*
- Lester, Al-Rfou, Constant (2021), *The Power of Scale for Parameter-Efficient Prompt Tuning*
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (revisited from [Lesson 2](../02-LoRA-and-QLoRA/README.md) for direct comparison)

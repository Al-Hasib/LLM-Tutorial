# Domain-Specific Fine-tuning Case Study

**Phase:** [Fine-tuning LLMs](../README.md) · **Topic folder:** `06-Domain-Specific-Finetuning-Case-Study`

## Why this matters

This lesson is the capstone of the phase: a single worked example that pulls together [Lesson 1's full-fine-tuning-vs-PEFT decision](../01-Full-Finetuning-vs-PEFT/README.md), [Lesson 2's LoRA mechanism](../02-LoRA-and-QLoRA/README.md), and [Lesson 5's real-world tooling](../05-Finetuning-with-HuggingFace-PEFT-TRL/README.md) into the question a real fine-tuning project actually starts with: *"I need this model to be good at my narrow domain — how do I do that, and what am I risking?"* Every other lesson in this phase examined one mechanism in isolation; this one runs an actual before/after comparison and reports what happened, including the parts that don't flatter either method.

## What this lesson covers

- Data curation considerations for a narrow domain
- Choosing full fine-tuning vs. PEFT for a domain-adaptation project, revisited concretely
- Why domain adaptation needs to be evaluated on *two* axes, not one: in-domain gain and general-capability regression
- A worked case study: adapting a small pretrained model to a distinctive "pirate speak" style with LoRA and with full fine-tuning, measured side by side

## 1. Data curation for a narrow domain

Fine-tuning for a narrow domain (a company's support-ticket style, a legal-document register, a specific writing voice) starts with the same data question [Lesson 4 §5](../04-Instruction-Tuning-SFT/README.md#5-data-quality-and-diversity-beat-raw-quantity) raised for instruction data generally, sharpened for the domain case: how narrowly should "narrow" be curated? Too little diversity within the domain (a handful of nearly-identical examples) risks the model memorizing surface patterns rather than the underlying style or vocabulary — exactly the gap [Lesson 4's own toy run](../04-Instruction-Tuning-SFT/README.md#6-what-examplepy-does) measured directly between training-set and held-out accuracy. Too much drift from the target domain in the training examples (mixed-quality scraped text, off-topic filler) dilutes the signal the fine-tuning run is meant to concentrate. In practice this means: collect examples that are unambiguously *representative* of the target domain's vocabulary and phrasing, hold out a genuinely separate slice for evaluation (never data the model trained on, exactly as `example.py` does below), and keep a separate general-purpose evaluation set on hand from the very start — you cannot measure forgetting after the fact if you didn't decide what "before" looked like in advance.

## 2. Full fine-tuning vs. PEFT for domain adaptation

[Lesson 1](../01-Full-Finetuning-vs-PEFT/README.md) laid out the abstract trade-off: full fine-tuning updates every parameter (highest ceiling, highest memory cost, highest forgetting risk) while PEFT methods like [LoRA](../02-LoRA-and-QLoRA/README.md) freeze the base and train a small add-on. Domain adaptation is the setting where that trade-off is most concrete, because the fine-tuning data is, almost by definition, narrower and less diverse than the pretraining corpus — precisely the condition under which full fine-tuning's unrestricted freedom to move every weight is most likely to overwrite general capability the base model already had, purely because nothing in the training data or loss discourages it from doing so. LoRA's frozen base is a structural constraint against that: whatever behavior isn't reachable by a rank-`r` update to a handful of projection matrices simply doesn't change, for better (general capability is protected) and for worse (the achievable domain-specific improvement has a ceiling the rank sets). `example.py` fine-tunes the *same* pretrained model both ways, on the *same* domain data, and measures where that trade-off actually landed for this run's real numbers.

## 3. Evaluating on two axes: specialization AND regression

The single most common mistake in domain fine-tuning write-ups is reporting only the metric that improved. A domain fine-tune that helps hugely on the target domain while quietly making the model worse at ordinary tasks is not obviously a good trade — whether it's worth it depends entirely on how the model will actually be used, and you can only make that call if you measured the regression at all. The methodology this lesson uses is simple and general enough to apply to any real domain-adaptation project:

1. Before touching the model, freeze two held-out evaluation sets: one **in-domain** (samples of the exact style you're targeting, never seen during fine-tuning) and one **general** (ordinary, broad-coverage examples of what the model could do before).
2. Measure both **before** fine-tuning — this is your baseline for catastrophic forgetting, not just your baseline for improvement.
3. Fine-tune.
4. Measure both again. Report the in-domain **improvement** and the general-capability **drift** together, not separately — a large improvement bought with a large regression is a materially different result from the same improvement bought with none.

`example.py` uses held-out **next-token-prediction loss** (lower = the model's learned distribution fits that text better) as the metric on both sides, since it's the one metric that applies uniformly to any text, in-domain or general, without needing a task-specific scorer.

## 4. What `example.py` does

1. Pretrains a small `MiniGPT` (the same block from [Phase 02 Lesson 6](../../Phase-02-Transformer-Architecture-Deep-Dive/06-Mini-Transformer-From-Scratch/README.md) and [Lesson 2](../02-LoRA-and-QLoRA/README.md)) on general, everyday-English sentences only.
2. Measures its held-out loss on general text and on a distinctly different "pirate speak" domain (`"arr the ship sails..."`, `"the cap'n found a chest of treasure"`) it has never seen — establishing the before-fine-tuning baseline on both axes.
3. Prints the trainable-parameter cost of full fine-tuning vs. LoRA for this exact model, using [Lesson 2's `LoRALinear`](../02-LoRA-and-QLoRA/README.md#5-what-this-lessons-code-does-and-what-a-real-workflow-uses-instead) — extended here with a `from_pretrained_linear` constructor so it wraps an *already-pretrained* layer's weights (the realistic case; Lesson 2's own demo wrapped a freshly-initialized one).
4. Fine-tunes two independent copies of the pretrained model on the same narrow pirate-speak training data: one with LoRA applied to the attention `W_q`/`W_v` projections only (everything else frozen), one with every parameter trainable.
5. Re-measures held-out loss on both the domain and general evaluation sets for both fine-tuned models, and reports the domain improvement and general-capability drift side by side — plus a qualitative generation from a domain-neutral prompt, to see whether pirate vocabulary leaks into text that gave no domain cue at all.

The conclusion printed at the end is derived directly from that run's real numbers — whichever way they land — rather than a pre-decided narrative, because the entire point of this lesson is that you can't know this trade-off in advance; you have to measure it.

## Video Script Outline

1. Motivation — "putting Lessons 1, 2, and 5 together on one real question: fine-tune for MY domain, at what cost?"
2. Data curation for a narrow domain: representative, diverse-enough, genuinely held-out
3. Full fine-tuning vs. LoRA, revisited specifically for the domain-adaptation setting
4. Why you must evaluate on two axes — in-domain gain and general-capability drift — not one
5. Walkthrough of `example.py`'s setup: pretraining, the two held-out sets, `LoRALinear.from_pretrained_linear`
6. Walkthrough of the LoRA vs. full-fine-tuning results: parameter cost, domain improvement, general-loss drift, and the qualitative generic-prompt check
7. Recap of the whole phase: PEFT/LoRA mechanism -> instruction tuning -> real tooling -> this case study tying it together
8. Preview: [Phase 06](../../Phase-06-Alignment-and-RLHF/README.md) picks up from a fine-tuned, instruction-following model and asks how to align its behavior with human preferences

## Further Reading

- Kirkpatrick et al. (2017), *Overcoming Catastrophic Forgetting in Neural Networks* (the general phenomenon measured directly here — first raised in [Lesson 1 §3](../01-Full-Finetuning-vs-PEFT/README.md#3-catastrophic-forgetting))
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (the method compared against full fine-tuning in this case study — full derivation in [Lesson 2](../02-LoRA-and-QLoRA/README.md))
- Gururangan et al. (2020), *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (domain-adaptive pretraining as a related, larger-scale strategy for the same underlying problem)
- Zhou et al. (2023), *LIMA: Less Is More for Alignment* (the data-curation angle applied to domain data in §1 — first covered in [Lesson 4](../04-Instruction-Tuning-SFT/README.md#5-data-quality-and-diversity-beat-raw-quantity))

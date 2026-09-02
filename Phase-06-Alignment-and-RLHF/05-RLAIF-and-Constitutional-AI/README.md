# RLAIF and Constitutional AI

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `05-RLAIF-and-Constitutional-AI`

## Why this matters

Every pipeline built so far in this phase shares one expensive ingredient: [Lesson 2&#39;s reward model](../02-Reward-Modeling/README.md) and both optimizers built on top of it ([Lesson 3&#39;s PPO](../03-RLHF-with-PPO/README.md) and [Lesson 4&#39;s DPO](../04-Direct-Preference-Optimization-DPO/README.md)) all assume a dataset of **human** pairwise preference labels. Collecting those labels at the scale modern LLMs need is slow and costly, and human labelers themselves get fatigued, disagree, and are limited in how many principles they can consistently apply across thousands of comparisons. **RLAIF** — Reinforcement Learning from AI Feedback (Bai et al., 2022; Lee et al., 2023) — asks whether a capable language model can generate those same preference judgments instead, at a fraction of the cost. This lesson keeps every piece of machinery from Lessons 2-4 exactly as it was; the only thing that changes is *who* produces the label "which response is better." **Constitutional AI** (Anthropic, Bai et al., 2022) is the fullest expression of this idea: it replaces not just the preference labeler but also the human-written SFT demonstrations, using a written **constitution** — an explicit list of principles — as the standard an AI model applies to critique and improve its own outputs. This directly sets up [Lesson 6](../06-Safety-Bias-and-Toxicity-Mitigation/README.md), where a written safety standard is applied systematically across a whole deployment pipeline rather than one training stage.

## What this lesson covers

- RLAIF: replacing the human preference labeler in Lesson 2's pipeline with an AI judge
- Constitutional AI's two phases: SL-CAI (self-critique and revision) and RL-CAI (AI preference labeling against a constitution)
- Why a *written* constitution, specifically, is what makes Constitutional AI's AI feedback more scalable and auditable than an unconstrained "ask a model if this is good"
- How SL-CAI's revised outputs become supervised fine-tuning data, and how RL-CAI's AI preference labels feed the exact same Bradley-Terry / DPO machinery from Lessons 2 and 4
- A hands-on demonstration: a rule-based critic (standing in for an LLM judge) that detects constitution violations, revises them, and measures the violation rate drop, plus a toy AI preference labeler choosing between original and revised outputs

## 1. RLAIF: an AI judge instead of a human labeler

The RLAIF pipeline is structurally identical to RLHF (Lesson 1 section 4's diagram): collect a prompt, generate a small number of candidate responses from the current policy, and produce a pairwise preference over them. RLHF asks a human to make that pairwise judgment; RLAIF instead prompts a separate (often larger or specially-instructed) language model with the two candidates and asks it to pick the better one, optionally with a rubric describing what "better" means for the task. That AI-generated preference label is then used exactly as a human label would be: to train a reward model with [Lesson 2&#39;s Bradley-Terry loss](../02-Reward-Modeling/README.md#3-the-reward-model-loss), or fed directly as `(chosen, rejected)` triples into [Lesson 4&#39;s DPO loss](../04-Direct-Preference-Optimization-DPO/README.md#3-the-dpo-loss). Lee et al. (2023) found AI-labeled preferences can match or exceed human-labeled RLHF on several tasks, at dramatically lower labeling cost — though the AI judge inherits whatever biases and blind spots the judging model itself has, which is a real limitation, not a footnote (Lesson 6 discusses bias measurement directly).

## 2. Constitutional AI, phase 1: SL-CAI (self-critique and revision)

Constitutional AI applies RLAIF's "use an AI instead of a human" idea to an entire alignment pipeline, not just the preference-labeling step, starting with the supervised stage. Given a **constitution** — a short list of explicit natural-language principles such as "choose the response that is least likely to encourage illegal or dangerous activity," or "choose the response that is more respectful and does not demean the user" — the model itself is used, in a loop, to:

```
1. Generate an initial response to a (possibly adversarial) prompt
2. CRITIQUE:  ask the model to identify ways the response violates a constitutional principle
3. REVISE:    ask the model to rewrite the response so it no longer violates that principle
4. Repeat over multiple principles / multiple critique-revise rounds if needed
```

The model is then **fine-tuned on its own revised outputs** — supervised fine-tuning exactly as in [Phase 05 Lesson 4](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md), except the `(prompt, good response)` pairs were produced by the model critiquing and correcting itself against a written standard, rather than by a human writing a demonstration from scratch. This is the "SL" in SL-CAI: **s**upervised **l**earning on **C**onstitutional-**AI**-generated data.

## 3. Constitutional AI, phase 2: RL-CAI (AI preference labeling against the constitution)

The second phase is RLAIF applied specifically with the constitution as the judging standard: instead of a human comparing two responses, the model itself (or a separate AI evaluator) is shown the constitution's principles and two candidate responses to the same prompt, and asked to pick which one better satisfies them. This produces exactly the pairwise `(chosen, rejected)` preference data [Lesson 2](../02-Reward-Modeling/README.md) is built around — the only difference from ordinary RLAIF (section 1) is that the judging standard is an explicit, written, auditable set of principles rather than an unspecified notion of "quality." Those AI-generated preferences then train a reward model (or feed DPO directly), and RL fine-tuning proceeds exactly as in [Lesson 3](../03-RLHF-with-PPO/README.md) — hence "RL-CAI." The full two-phase pipeline (SL-CAI, then RL-CAI) is what Anthropic used to train Claude to be more harmless without needing human labelers to read and judge harmful content directly, which was also an explicit goal: reducing the amount of disturbing content human raters are exposed to during training.

## 4. Why a written constitution specifically

The constitution is not just a prompt-engineering trick — it is what makes AI feedback **scalable and auditable** in a way that an unconstrained "is this response good?" query is not: every principle is stated explicitly, can be individually inspected, revised, added, or removed by the people building the system, and every critique or preference judgment can in principle be traced back to which specific principle motivated it. This is a meaningfully different design point from RLAIF in general (section 1), which can use an arbitrary, possibly implicit standard of "better" baked into the judge model's own training. `example.py` makes this concrete with a small, explicit, inspectable set of principles — deliberately simplified into rule-based pattern checks rather than an LLM judge, so the mechanics of "critique against a written standard, then revise, then measure the effect" are fully transparent.

## 5. Honest limitations of this lesson's toy implementation

`example.py`'s critic and AI preference labeler are **rule-based pattern matchers**, not language models — they can only detect the exact textual patterns they were written to catch (a specific insult word, a specific unsafe chemical combination, a specific overconfidence phrase). A real Constitutional AI system uses an actual LLM to read and judge arbitrary text against a principle stated in natural language, which generalizes far beyond any fixed set of regexes. The toy version here should be read as a fully transparent, inspectable *stand-in* for that judgment call — useful for seeing the two-phase mechanics end to end, not as a realistic safety classifier (see [Lesson 6](../06-Safety-Bias-and-Toxicity-Mitigation/README.md) for a more careful treatment of toxicity classification specifically).

## Video Script Outline

1. Motivation — human preference labeling is a bottleneck; can a model provide the labels instead?
2. RLAIF: the same pairwise-preference pipeline as RLHF, with an AI judge substituted for the human labeler (section 1)
3. Constitutional AI phase 1 (SL-CAI): self-critique and revision against a written constitution, then fine-tuning on the revised outputs (section 2)
4. Constitutional AI phase 2 (RL-CAI): AI preference labeling against the same constitution, feeding straight into Lesson 2/4's machinery (section 3)
5. Why an explicit, written constitution is more scalable and auditable than an unconstrained AI judgment (section 4)
6. Walkthrough of `example.py`'s rule-based constitution: the three toy principles and their check/revise functions
7. Live results: violation rate before vs after critique-and-revise, and the AI preference labeler's choices between original and revised outputs
8. Honest caveat: this is a rule-based stand-in for an LLM critic, and a preview of Lesson 6's bias/toxicity measurement

## Further Reading

- Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*
- Bai et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (the RLHF baseline Constitutional AI compares against and builds on)
- Lee et al. (2023), *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*
- Ganguli et al. (2022), *Red Teaming Language Models to Reduce Harms* (Anthropic's red-teaming methodology, referenced further in Lesson 6)

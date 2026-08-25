# Chain-of-Thought and Reasoning Prompts

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `02-Chain-of-Thought-and-Reasoning-Prompts`

## Why this matters

[Lesson 1](../01-Prompting-Basics-Zero-Few-Shot/README.md) established that a frozen model's only lever is the prompt, and measured how the *number* and *order* of in-context examples change accuracy. This lesson asks a different question: for problems that need more than one step of reasoning (arithmetic word problems, multi-hop logic), can the *content* of the prompt — specifically, asking the model to show intermediate work — change accuracy on its own, without adding a single new example? And once a model can produce a reasoning trace at all, can you extract more reliability out of it for free, just by sampling it several times? That second question has a clean, provable answer, and it is the heart of this lesson's `example.py`. Both ideas are prerequisites for [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md), which generalizes "one reasoning chain" into "a searchable tree of reasoning branches."

## What this lesson covers

- Chain-of-Thought (CoT) prompting: eliciting intermediate reasoning steps via few-shot worked examples
- Zero-shot CoT: the "Let's think step by step" trick, no worked examples required
- Self-consistency: sampling multiple independent reasoning paths and majority-voting the final answer
- The Condorcet Jury Theorem as the mathematical backbone of why self-consistency works
- `example.py`: an exact-formula-plus-simulation proof of self-consistency's benefit — and of the honest case where it backfires
- Why fragmenting wrong answers (vote-splitting) makes self-consistency even more effective on real, open-ended tasks

## 1. Chain-of-Thought prompting (Wei et al., 2022)

Standard few-shot prompting (Lesson 1) shows the model `(input, final_answer)` pairs. **Chain-of-Thought (CoT)** prompting instead shows `(input, reasoning_steps, final_answer)` triples — the few-shot examples themselves demonstrate *working through* the problem, not just stating the answer:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls is 6 balls. 5 + 6 = 11. The answer is 11.

Q: <new question>
A:
```

Wei et al. (2022) showed that on multi-step arithmetic, commonsense, and symbolic reasoning benchmarks, this single change — demonstrating *how* to reason, not just *what* the answer is — produces large accuracy jumps on sufficiently large models, with negligible effect on small ones. This is an **emergent capability** in the sense used in [Phase 03 Lesson 1's Further Reading](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#further-reading): the benefit of CoT prompting is barely present below a certain scale and becomes large above it, without any change to the prompting technique itself.

## 2. Zero-shot Chain-of-Thought (Kojima et al., 2022)

Writing full worked-example reasoning traces by hand for every task is expensive. Kojima et al. (2022) showed a remarkably simple alternative: append the literal phrase **"Let's think step by step"** after the question, with *no* worked examples at all, and let the model generate its own reasoning trace from scratch before answering. This recovers a substantial fraction of few-shot CoT's benefit on the same benchmarks, purely from one fixed instruction string. It is a striking demonstration of how much latent reasoning capability scale unlocks even without any demonstrations — the model already "knows how" to decompose the problem; it mostly needs to be told to actually do so before committing to a final answer, rather than answering immediately.

## 3. Self-consistency (Wang et al., 2022): sample many, vote

CoT and zero-shot CoT both produce ONE reasoning trace, typically via greedy or low-temperature decoding. Wang et al. (2022) observed that different reasoning paths can arrive at the same problem from different angles, and don't all fail in the same way — so instead of decoding once, **self-consistency**:

1. Samples `k` independent reasoning traces for the *same* question (via temperature/top-p sampling, so each trace can genuinely differ);
2. Extracts each trace's final answer;
3. Returns the **majority-vote** answer across all `k` traces, discarding the individual reasoning traces entirely.

This costs `k` times the inference compute of a single sample, in exchange for higher accuracy — a compute/accuracy trade-off, not a free lunch. Understanding exactly how much accuracy that compute buys, and under what conditions it can backfire, is what `example.py` proves directly.

## 4. The Condorcet Jury Theorem: why voting works — and when it doesn't

Model one reasoning sample abstractly: it lands on the correct final answer with probability `p`, independently of every other sample. This is precisely the setup of the **Condorcet Jury Theorem** (1785): given `k` independent "voters" each correct with probability `p`, and majority rule,

- if `p > 0.5`, the probability the **majority** is correct increases monotonically towards **1** as `k -> infinity`;
- if `p < 0.5`, the probability the majority is correct decreases monotonically towards **0** as `k -> infinity`;
- if `p == 0.5` exactly, majority accuracy stays at exactly **0.5** for every `k` — there is no signal to amplify.

`example.py` computes the exact binomial formula for "probability that more than `k/2` of `k` independent Bernoulli(p) trials are correct," validates it against an independent Monte Carlo simulation, and then sweeps both `k` and `p` to display all three regimes numerically. The `p < 0.5` regime is not a hypothetical edge case worth skipping — it is the honest, necessary flip side of the theorem, and it means self-consistency is only a good idea when there is independent reason to believe the base reasoning method already beats chance (`p > 0.5`) on the task at hand; applying it blindly to a method that is *worse* than chance makes the final answer reliably worse, not better.

## 5. Beyond binary: vote-splitting makes real self-consistency even stronger

The Condorcet model above assumes exactly two possible answers, "correct" and "the wrong answer" — a single unified opposition. Real CoT tasks (arithmetic, multi-hop QA) have open-ended answer spaces: a numeric answer, a free-text span. Two different flawed reasoning paths rarely land on the *exact same* wrong number. `example.py`'s final experiment spreads the `(1 - p)` probability mass across several distinct wrong labels instead of one, and shows that **plurality voting** over this fragmented answer space recovers correct-answer accuracy even in some regimes where the binary-case formula alone would look bleak, purely because the wrong votes are split against each other (classic vote-splitting) rather than consolidated into a single competing bloc. This is one honest, mechanistic reason self-consistency's reported real-world gains can exceed what the plain binary Condorcet model alone would predict — while still respecting the same underlying limit: if the wrong answers are numerous enough and `p` low enough, splitting them thinner is not enough to save the vote.

## Video Script Outline

1. Motivation — CoT changes *what's in* the prompt's reasoning, not just how many examples; self-consistency then squeezes more reliability out of that reasoning for a compute cost
2. Chain-of-Thought (Wei et al. 2022): show the worked-example prompt format, and the emergent-with-scale result
3. Zero-shot CoT (Kojima et al. 2022): "Let's think step by step" as a one-line trick
4. Self-consistency (Wang et al. 2022): sample k reasoning traces, majority-vote the final answers
5. The Condorcet Jury Theorem: derive why voting helps when p > 0.5
6. Walkthrough of `example.py`'s exact-formula + Monte Carlo proof — the helpful case, the p=0.5 dead zone, and the honest p < 0.5 failure case
7. The vote-splitting extension: why real, open-ended-answer self-consistency can do even better than the binary math suggests
8. Recap + preview: turning one reasoning chain into a searchable tree of many (Tree-of-Thought, Lesson 3)

## Further Reading

- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners*
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- de Condorcet (1785), *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix* (the original Jury Theorem)
- Wei et al. (2022), *Emergent Abilities of Large Language Models* (revisited from [Phase 03 Lesson 1](../../Phase-03-LLM-Architectures-and-Types/01-Decoder-Only-Models-GPT-Family/README.md#further-reading); explains why CoT's benefit is scale-dependent)

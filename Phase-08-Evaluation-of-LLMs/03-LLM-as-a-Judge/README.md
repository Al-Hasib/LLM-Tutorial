# LLM-as-a-Judge

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `03-LLM-as-a-Judge`

## Why this matters

[Lesson 1](../01-Evaluation-Metrics/README.md) ended on an uncomfortable, concrete result: a genuinely correct paraphrase scored *worse* on BLEU/ROUGE than a fluent answer that got a key fact wrong, because those metrics only count shared n-grams and have no notion of meaning. [Lesson 2](../02-Standard-Benchmarks/README.md) showed the field's other main tool — standardized benchmarks — works only because each question has one unambiguous correct answer to check a log-probability against. Neither tool has any way to grade open-ended generation: "write a helpful, well-reasoned answer to this customer's question" has no reference string to overlap with and no single correct token to score. **LLM-as-a-Judge** is the field's answer to that gap: use a strong LLM itself to read a candidate response and grade it — for meaning, helpfulness, correctness — the same way a human evaluator would, but at a fraction of the cost and time. It is also the direct evaluation backbone of preference-based training you'll meet in [Phase 06 Lesson 2 (Reward Modeling)](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md): a reward model *is* a learned judge, trained on the same kind of pairwise preference data an LLM judge (or a human, see [Lesson 5](../05-Human-Evaluation-Methodologies/README.md)) produces. This lesson covers the two ways to ask a judge for a verdict, the systematic biases that make a naive judge's verdict untrustworthy, and the mitigations that make it usable in practice.

## What this lesson covers

- Two judge protocols: pairwise comparison ("which response is better") and absolute rubric scoring ("rate this response 1-10 on these criteria")
- Position bias — a judge's tendency to favor whichever response it sees first (or second)
- Verbosity bias — a judge's tendency to reward length regardless of quality
- Self-preference bias — a judge's tendency to favor outputs that resemble its own family's style
- Standard mitigations: swap-and-average orderings, length-controlling the score, using a diverse panel of judges
- A from-scratch simulation quantifying exactly how much these biases distort measured win rates, and how much each mitigation recovers

## 1. Two judge protocols

**Pairwise comparison.** Show the judge a prompt and two candidate responses (from two different models, or two different decoding runs of the same model), and ask which one is better — optionally allowing a "tie." The output is a single relative preference. Aggregated over many prompts, pairwise judgments produce a **win rate** between two systems, and aggregated over many systems they can feed an Elo-style ranking exactly like the one built from scratch in [Lesson 5](../05-Human-Evaluation-Methodologies/README.md#2-elo-rating-from-pairwise-comparisons). This is the protocol behind public arenas like Chatbot Arena.

**Absolute (rubric) scoring.** Show the judge one response at a time, plus a rubric describing what to grade (e.g., "rate helpfulness, correctness, and clarity each from 1-5"), and ask for a score per axis. This produces an absolute number per response instead of a relative preference, which is useful when you need per-response diagnostics rather than just "which is better," but it is generally noisier and less consistent across a judge's own repeated calls than a direct A-vs-B comparison, because "how good is a 7 out of 10" has no anchor the way "is A better than B" does.

Both protocols share the same underlying assumption: that a sufficiently strong LLM's judgment correlates well with what a careful human would say. That assumption holds reasonably well in aggregate (Zheng et al., 2023 report ~80%+ agreement between GPT-4 judgments and human preferences on their MT-Bench setup) but breaks down in specific, systematic, and importantly *predictable* ways — which is the entire subject of the rest of this lesson.

## 2. Position bias

Given the exact same pair of responses, a pairwise judge's verdict can flip depending on which response is shown **first** versus **second** in the prompt — independent of quality. This is **position bias**, and it is large enough in practice that early GPT-4-as-judge studies found the judge agreed with itself only 60-80% of the time across the two orderings of the identical pair (Wang et al., 2023; Zheng et al., 2023). Concretely: if a naive evaluation harness always places the new/candidate model's output in the first slot and the baseline's output in the second slot (a very natural thing to do when writing an eval script), any nonzero position bias directly and silently inflates the candidate's measured win rate — regardless of whether the candidate is actually any better. `example.py` reproduces exactly this failure with real numbers.

## 3. Verbosity bias

A judge — like many human readers — tends to associate length with thoroughness, and rewards longer responses even when the extra length adds no correctness or usefulness (Zheng et al., 2023; Dubois et al., 2024). This is dangerous specifically because it is easy to exploit: a model fine-tuned or prompted to simply *write more* can look like it improved, on a judge's scorecard, without having gotten any better at the underlying task. Length-controlled evaluation setups (e.g., AlpacaEval's "length-controlled win rate," Dubois et al., 2024) exist specifically to strip this effect back out of a measured score.

## 4. Self-preference bias

A judge tends to rate outputs that resemble its own model family's style, phrasing conventions, or reasoning pattern more favorably — even when an independent human wouldn't share that preference (Zheng et al., 2023 call this "self-enhancement bias," e.g. GPT-4 as judge modestly favoring GPT-4-style outputs). This is the most insidious of the three biases because it is invisible from a single-judge study: everything looks consistent and confident, it's just consistently tilted toward one family's house style.

## 5. Mitigations

Each bias above has a corresponding, practical fix, all demonstrated with real before/after numbers in `example.py`:

- **Position bias -> evaluate both orderings.** Run the comparison twice per pair — once with A first, once with B first — and either (a) average the two resulting preference scores (this makes any *constant* position bonus cancel out algebraically, since it is added to different responses in the two runs) or (b) only accept a verdict where the judge agrees with itself across both orderings, treating disagreement as a tie. Either approach costs exactly 2x the judge calls and removes the bias's effect on the *aggregate* win rate.
- **Verbosity bias -> control for length.** Either instruct the judge explicitly to ignore length, or — more robustly — measure the judge's own revealed length preference (e.g., by regressing judge scores against response length across many graded pairs) and subtract that estimated effect back out of each score before comparing. This is the same idea behind AlpacaEval's length-controlled win rate.
- **Self-preference bias -> use a diverse panel of judges.** No single judge is unbiased, but if several judges from *different* model families each have their own, roughly independent house-style preference, averaging their verdicts (a "jury") suppresses each individual judge's idiosyncratic bias in the aggregate, the same statistical logic as averaging any noisy, differently-biased estimators.

None of these mitigations make a judge perfect — they reduce a *systematic, measurable* distortion, they do not add missing judgment ability. A judge (LLM or human) can still be wrong about the actual quality of a response even after every bias correction above; these fixes only remove errors that come from *how the comparison was set up*, not errors in the judge's underlying competence.

## Video Script Outline

1. Motivation — Lessons 1-2 showed exactly what's missing: a way to score open-ended, reference-free generation
2. LLM-as-a-Judge: use a strong model to grade another model's output, pairwise vs. absolute scoring
3. Position bias: same pair, same quality, different verdict depending on slot order
4. Verbosity bias: length rewarded independent of actual quality, and why that's exploitable
5. Self-preference bias: a judge favoring its own family's style, invisible from a single-judge study
6. The three mitigations: swap-and-average, length-control, diverse judge panels
7. Walkthrough of `example.py` — a toy biased judge, quantified bias effects, and quantified mitigation recovery, all with real numbers
8. Recap + pointer to [Lesson 5](../05-Human-Evaluation-Methodologies/README.md), where the same pairwise-preference idea is aggregated into Elo ratings — this time from human votes

## Further Reading

- Zheng, Chiang, Sheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (the paper that named and measured position, verbosity, and self-enhancement bias in LLM judges)
- Wang, Yuan, Yao et al. (2023), *Large Language Models are not Fair Evaluators* (a focused study of position bias in pairwise LLM judging, and calibration fixes)
- Dubois, Galambosi, Liang, Hashimoto (2024), *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators*
- Zeng, Attia, Wu et al. (2024), *Evaluating Large Language Models at Evaluating Instruction Following* (LLMBar — a benchmark specifically designed to stress-test judge robustness to superficial cues like length and style)

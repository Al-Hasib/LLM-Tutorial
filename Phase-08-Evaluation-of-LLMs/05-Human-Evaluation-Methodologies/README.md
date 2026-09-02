# Human Evaluation Methodologies

**Phase:** [Evaluation of LLMs](../README.md) · **Topic folder:** `05-Human-Evaluation-Methodologies`

## Why this matters

[Lesson 3](../03-LLM-as-a-Judge/README.md) used an LLM to stand in for a human evaluator, precisely because a real human panel is slow and expensive to run at scale — but the LLM judge's whole design (pairwise comparison, rubric axes, the biases to correct for) was modeled directly on how human evaluation actually works. This lesson covers the original, gold-standard version: how you design a rubric, how you turn a pile of pairwise human preference votes into a single ranking, and — critically — how you check whether your human annotators even agree with each other before trusting their votes at all. The rubric axes this lesson uses (helpfulness, accuracy, harmlessness) are the direct operationalization of the **HHH** framing from [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing), and the "accuracy" axis is graded using exactly the entailment judgment built in [Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md#3-automated-factuality-checking-two-complementary-approaches). The pairwise preference data collected here is also the raw material [Phase 06 Lesson 2 (Reward Modeling)](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md) trains a reward model on — human evaluation isn't just how you check a model after the fact, it's the data source the entire RLHF pipeline runs on.

## What this lesson covers

- Rubric design: multiple evaluation axes instead of one overall "good/bad" score
- Pairwise comparison as the human-annotation protocol of choice, and why
- Elo rating: the chess rating system, borrowed wholesale to rank models from pairwise votes
- Inter-annotator agreement: why raw percent-agreement is misleading
- Cohen's kappa: the formula that corrects percent-agreement for chance

## 1. Rubric design: multiple axes, not one score

Asking a human annotator "is this response good?" collapses several genuinely independent questions into one number, which hides exactly the information you need to diagnose *why* a model is bad. Standard practice instead asks annotators to rate several axes separately, most commonly a version of the same **HHH** triad from [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing):

- **Helpfulness** — does the response actually address the user's request?
- **Accuracy** — are the response's factual claims correct and supported (the human version of [Lesson 4](../04-Hallucination-and-Factuality-Evaluation/README.md)'s entailment check)?
- **Harmlessness** — does the response avoid unsafe, offensive, or otherwise harmful content?

Separating these matters because they can — and often do — move independently: a response can be extremely helpful and accurate while being subtly unsafe (giving correct, working instructions for something dangerous), or maximally harmless while being useless (refusing a benign request). A single overall score averages these tensions away; a multi-axis rubric keeps them visible, exactly the same tension [Phase 06 Lesson 1](../../Phase-06-Alignment-and-RLHF/01-The-Alignment-Problem/README.md#3-the-hhh-framing) already flagged between the three H's.

## 2. Elo rating from pairwise comparisons

Rather than asking annotators to assign an absolute score to a single response (noisy and hard to calibrate — see [Lesson 3 §1](../03-LLM-as-a-Judge/README.md#1-two-judge-protocols)), the dominant human-evaluation protocol is **pairwise comparison**: show two models' responses to the same prompt, ask which is better (or tie), and repeat across many prompts and many pairs of models. The question then becomes: given a large pile of pairwise win/loss/tie results, how do you turn that into a single ranked list? This is exactly the problem chess had already solved.

**The Elo rating system** (Arpad Elo, developed for chess, adapted directly for model ranking by public leaderboards like Chatbot Arena): every competitor (here, every model) has a rating `R`. Given two competitors with ratings `R_A` and `R_B`, the *expected* probability that A beats B is a logistic function of the rating gap:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

Note `E_A + E_B = 1` (they're complementary), and if `R_A == R_B` then `E_A = 0.5` — equal ratings predict a coin-flip outcome, exactly as expected. After an actual match with observed outcome `S_A` (`S_A = 1` if A won, `0` if A lost, `0.5` for a tie), both ratings are updated by the **surprise** — the gap between what actually happened and what was expected:

```
R_A_new = R_A + K * (S_A - E_A)
R_B_new = R_B + K * (S_B - E_B)
```

`K` is a fixed step size controlling how fast ratings move per match (a larger `K` adapts faster but is noisier). Read the update rule directly: if A was expected to win with `E_A = 0.9` and actually won (`S_A = 1`), the surprise `S_A - E_A = 0.1` is small, so A's rating barely moves — beating a much weaker opponent is not informative. If A was expected to *lose* (`E_A = 0.1`) but won anyway, the surprise `S_A - E_A = 0.9` is large, and A's rating jumps — an upset is highly informative about A actually being stronger than its current rating reflects. Run this update over thousands of pairwise match results (exactly how Chatbot Arena aggregates millions of human votes into its public leaderboard) and, as `example.py` demonstrates with synthetic matches between models of known true quality, the ratings converge to correctly rank the competitors even though every individual match result is noisy.

## 3. Inter-annotator agreement and why raw percent-agreement is misleading

Before trusting any human-annotated dataset (rubric scores or pairwise votes), you need to know whether independent annotators actually agree with each other — if they don't, the "ground truth" labels are closer to noise than signal. The naive approach is **percent agreement**: the fraction of items where two annotators gave the same label. This is misleading for a specific, well-understood reason: **it never corrects for the agreement you'd expect from pure chance**, and chance-agreement can be very high whenever labels are imbalanced. If 95% of responses in a safety-labeling task are "safe" and only 5% are "unsafe," two annotators who each just guess "safe" every single time will agree on 95%+ of items — a percent-agreement score that looks excellent while reflecting *zero* actual judgment.

## 4. Cohen's kappa

**Cohen's kappa** (Cohen, 1960) fixes this by explicitly subtracting out chance agreement:

```
kappa = (p_o - p_e) / (1 - p_e)
```

- `p_o` — **observed agreement**: the raw fraction of items where the two annotators gave the same label (this is exactly the naive percent-agreement from §3).
- `p_e` — **expected agreement by chance**: the agreement rate two annotators would produce if each was independently guessing according to their own observed label distribution. Computed as `p_e = sum_k( P1(k) * P2(k) )`, where `P1(k)` and `P2(k)` are the fraction of items each annotator labeled as class `k` — i.e., the probability both would independently pick class `k` by chance, summed over all classes.

Read the formula directly: `p_o - p_e` is how much agreement was achieved *above and beyond* chance, and dividing by `1 - p_e` (the maximum possible agreement beyond chance) normalizes that into a 0-to-1-ish scale. `kappa = 1` means perfect agreement; `kappa = 0` means the annotators agree exactly as often as chance alone would predict (their apparent agreement carries no real signal); `kappa < 0` means they agree *less* than chance would predict. This is exactly why kappa, not percent agreement, is the standard reported statistic for inter-annotator reliability in any serious human-evaluation study — a high `p_o` with a high `p_e` (as in the imbalanced-label example above) collapses to a low or near-zero kappa, correctly revealing that the raw agreement number was an illusion of the label imbalance rather than real annotator consensus.

## 5. What `example.py` demonstrates

Elo ratings are implemented from scratch and run over a long sequence of synthetic pairwise "human preference" match results between several toy models with different, hidden true quality levels — showing the resulting ratings converge to the correct quality ranking despite every individual match being a noisy coin flip. Cohen's kappa is then implemented from scratch and computed on two synthetic annotator label sets in two contrasting scenarios: a genuine high-agreement case, and an imbalanced-label case where raw percent agreement looks high but kappa correctly reveals it is mostly a chance/imbalance artifact.

## Video Script Outline

1. Motivation — human evaluation is the gold standard [Lesson 3](../03-LLM-as-a-Judge/README.md)'s LLM judge was modeled on, and the direct data source for [Phase 06&#39;s reward models](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md)
2. Rubric design: helpfulness / accuracy / harmlessness as independent axes, echoing HHH
3. Pairwise comparison as the preferred annotation protocol, and why it beats absolute scoring
4. Elo rating: the chess formula, the expected-score function, and the "surprise"-driven update rule
5. Walkthrough of `example.py` part 1 — synthetic matches between models of known quality, watch Elo ratings converge to the true ranking
6. Why percent agreement is misleading: the imbalanced-label illusion
7. Cohen's kappa: the formula, and what p_o, p_e, and the normalization actually mean
8. Walkthrough of `example.py` part 2 — kappa correctly distinguishing real agreement from chance-driven agreement, with real numbers

## Further Reading

- Elo, A. (1978), *The Rating of Chessplayers, Past and Present* (the original Elo rating system)
- Cohen, J. (1960), *A Coefficient of Agreement for Nominal Scales* (the original Cohen's kappa paper)
- Chiang, Zheng, Sheng et al. (2024), *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* (Elo-style ranking applied to LLMs at scale, from millions of pairwise human votes)
- Bai, Jones, Ndousse et al. (2022), *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* (the human-preference data collection pipeline underlying reward modeling, see [Phase 06 Lesson 2](../../Phase-06-Alignment-and-RLHF/02-Reward-Modeling/README.md))
- Artstein, Poesio (2008), *Inter-Coder Agreement for Computational Linguistics* (a thorough survey of kappa and related agreement statistics for NLP annotation)

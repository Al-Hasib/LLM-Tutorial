# Reasoning Models and GRPO

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `07-Reasoning-Models-and-GRPO`

## Why this matters

[Lesson 3 (PPO)](../03-RLHF-with-PPO/README.md) and [Lesson 4 (DPO)](../04-Direct-Preference-Optimization-DPO/README.md) both optimize a policy against a signal derived from *human preference* — a learned reward model, or DPO's implicit reward from preference pairs. Neither ever checks anything against a ground truth; both only ever ask "which response did a human (or a model trained to imitate one) prefer?" Modern reasoning models (OpenAI's o1, DeepSeek-R1) train with RL against a fundamentally different kind of reward: a **verifiable** one — is the final numeric answer correct, did the generated code pass its unit tests — computed by a deterministic checker, not a learned model, and immune to the reward-hacking failure mode [Lesson 3 section 3](../03-RLHF-with-PPO/README.md#3-why-plain-policy-gradient-breaks-reward-hacking) had to guard against with a KL penalty. This lesson covers that shift, the RL algorithm (**GRPO**) that made training on it cheap enough to run at scale by removing PPO's critic network entirely, and a second, complementary lever this unlocks: spending more *compute at inference time* — directly extending [Phase 07 Lesson 2](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md)'s self-consistency idea from "a prompting trick for a frozen model" to "the deliberate design point of how these models are meant to be used."

## What this lesson covers

- Verifiable rewards vs. learned/preference-based rewards (Lessons 2-4's world vs. this lesson's)
- GRPO (Shao et al., 2024 — DeepSeekMath): PPO without a critic — group-relative advantage estimation
- Two properties of that group-relative advantage, measured directly: a real "push away from failure" signal, and a consistent signal scale across easy/medium/hard prompts
- A real, runnable GRPO training loop on a toy verifiable-reward task — and an honest look at what this specific toy task does and doesn't prove
- Test-time scaling: spending inference compute (more samples, majority vote) as a second axis distinct from [Phase 03 Lesson 5](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md)'s train-time scaling laws
- A real experiment: majority-vote accuracy vs. number of samples, on a deliberately still-imperfect policy

## 1. Two kinds of reward

Recall [Lesson 2](../02-Reward-Modeling/README.md)'s reward model `r_phi(x, y)`: a neural network trained to imitate human pairwise preferences, then used as a proxy for "how good is this response." A **verifiable reward** replaces that proxy with a cheap, deterministic function of the completion and a known correct answer — e.g. `reward = 1 if extracted_final_answer == ground_truth else 0` for a math problem, or `reward = fraction of unit tests passed` for generated code. No human labels are needed for the reward *itself* at RL time (labels were only needed earlier, to know the correct answers or write the tests) — and critically, this reward can't be gamed the way a learned reward model can ([Lesson 3 section 3](../03-RLHF-with-PPO/README.md#3-why-plain-policy-gradient-breaks-reward-hacking)'s reward hacking), because it is checking an actual verifiable fact, not a proxy model's opinion.

## 2. GRPO: PPO without a critic

Recap PPO's ingredients ([Lesson 3 section 5](../03-RLHF-with-PPO/README.md#5-ppos-clipped-surrogate-objective)): a policy, a value/critic network estimating a per-state baseline for the advantage, and the clipped surrogate objective. That critic is roughly as expensive to train and run as the policy itself. GRPO's fix (Shao et al., 2024, *DeepSeekMath*): for each prompt, sample a **group** of `G` completions from the current policy, score each with the verifiable reward, and use the **group's own mean and standard deviation** to turn each completion's reward into an advantage directly — no critic network anywhere:

```
advantage_i = ( reward_i - mean(rewards in group) ) / ( std(rewards in group) + eps )
```

The rest is identical to Lesson 3: the same clipped-surrogate-per-token objective (reusing that lesson's `L_CLIP` structure), and the same KL penalty against a frozen reference policy — just with this group-relative advantage in place of a learned baseline.

## 3. Why the group-relative baseline works — two properties, measured directly

A good RL baseline only needs to reduce variance without introducing bias — subtracting *any* quantity that doesn't depend on the sampled action satisfies that, and the group mean, computed fresh per-prompt from `G` real rollouts of that exact prompt, is exactly such a quantity. `example.py` measures two concrete consequences of this, independent of any specific training run:

- **A real negative-advantage signal.** With no baseline, a failed completion (`reward = 0`) gets advantage exactly `0` — it is never actively pushed down, only successes get pushed up. GRPO's group-relative advantage is *negative* for any below-group-average completion, which includes every failure whenever *any* group member succeeded — a genuine "move away from this" signal plain reward-only REINFORCE structurally cannot provide.
- **A consistent signal scale, regardless of task difficulty.** A raw, un-normalized reward's variance is `p(1-p)` where `p` is the per-sample success probability — this *vanishes* as `p` approaches 0 or 1, meaning a very easy or very hard prompt gives almost no gradient signal at all, even though there may be plenty left to learn (e.g. going from 95% to 99.9% success). GRPO's z-scoring keeps the advantage's scale close to constant across the entire difficulty range, because it always renormalizes against that specific prompt's own observed spread.

## 4. `example.py` Part A — the mechanism, then an honest end-to-end run

The toy task: `NUM_PROMPTS` distinct prompts, each asking for `SEQ_LEN` digits (0-5) that sum to a specific target — a small but genuinely verifiable correctness check, not a fabricated exploitable reward like [Lesson 3](../03-RLHF-with-PPO/README.md)'s deliberately-hackable "hack token." Part A runs three things:

1. The negative-advantage-fraction measurement above, on one rollout from a near-random policy — with real printed numbers.
2. The signal-scale-consistency measurement above, swept across simulated success probabilities from 3% to 97% — with real printed numbers.
3. An actual end-to-end training run of the full GRPO update (group-relative advantage + clipped surrogate + KL penalty), alongside the no-baseline ablation, on the SAME task.

Section 3 comes with an honest caveat printed directly by the script: on this specific tiny, low-dimensional toy search space (216 possible digit sequences per prompt), *both* methods end up solving the task, because Adam's own adaptive step sizing partly compensates for raw reward's inconsistent scale here. That is a property of this particular toy example, not evidence that baselines don't matter — sections 1-2 measure the actual mechanism GRPO provides directly, and that mechanism is what separates "learns from sparse, verifiable rewards at all" from "doesn't" once the search space is a real space of many-step token sequences, not 216 three-digit combinations.

## 5. Test-time scaling: a second lever, orthogonal to train-time scaling

[Phase 03 Lesson 5](../../Phase-03-LLM-Architectures-and-Types/05-Scaling-Laws/README.md) covers scaling *train-time* compute — more parameters, more data, more training FLOPs — to reduce loss. Reasoning models add a second, independent lever: spend more compute *at inference time*, on a fixed, already-trained model, to get a better answer for one query — e.g. sampling `N` completions and taking a majority vote. This is exactly [Phase 07 Lesson 2 section 3](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#3-self-consistency-wang-et-al-2022-sample-many-vote)'s self-consistency mechanism, here framed as the deliberate design point of a reasoning model rather than a prompting trick bolted onto an ordinary one.

## 6. `example.py` Part B — majority-vote accuracy vs. N

A policy already at ~100% one-shot accuracy has nothing left for majority voting to improve, so Part B trains a **fresh** policy with GRPO but stops it deliberately early — after a small fraction of the iterations Part A uses — leaving it genuinely imperfect. Using that partially-trained policy, unchanged, Part B samples `N` completions per prompt, extracts each one's digit-sum as its "answer," and takes the majority-vote answer across all `N`, for `N` in `{1, 3, 5, 9, 15, 25}`, over many independent trials. The real, printed accuracy climbs substantially from `N=1` to `N=25`, with the per-sample rate of improvement tapering off as `N` grows — the same diminishing-returns shape [Phase 07 Lesson 2 section 4](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#4-the-condorcet-jury-theorem-why-voting-works--and-when-it-doesnt)'s Condorcet Jury Theorem predicts for any per-sample accuracy above chance. Because a wrong digit-sequence can land on several different incorrect sums rather than one single competing answer, this is really the multi-candidate plurality-voting regime from [Lesson 2 section 5](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md#5-beyond-binary-vote-splitting-makes-real-self-consistency-even-stronger), not the plain binary case — wrong votes split against each other across several wrong sums, part of why the correct answer wins more easily than the raw per-sample accuracy alone would suggest.

## Video Script Outline

1. Motivation — Lessons 3-4 optimize against human preference; reasoning models optimize against a verifiable, checkable fact instead
2. Verifiable rewards vs. learned reward models, and why the former can't be reward-hacked the way the latter can
3. GRPO: remove PPO's critic, replace it with a group-relative, per-prompt baseline computed from real rollouts
4. Walkthrough of `example.py` Part A demo 1 — the negative-advantage-fraction measurement, real numbers
5. Walkthrough of `example.py` Part A demo 2 — signal-scale consistency across easy/medium/hard prompts, real numbers
6. Walkthrough of `example.py` Part A demo 3 — the full GRPO update training end to end, and the honest caveat about this toy task's scale
7. Test-time scaling as a second, inference-time lever, distinct from Phase 03 Lesson 5's train-time scaling laws
8. Walkthrough of `example.py` Part B — majority-vote accuracy vs. N on a deliberately imperfect policy, tying back to Phase 07 Lesson 2's self-consistency and vote-splitting

## Further Reading

- Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo (2024), *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (introduces GRPO)
- DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- OpenAI (2024), *Learning to Reason with LLMs* (the o1 announcement, describing RL training against verifiable outcomes)
- Snell, Lee, Xu, Kumar (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models* — already cited in [Phase 07 Lesson 2](../../Phase-07-Prompt-Engineering-and-In-Context-Learning/02-Chain-of-Thought-and-Reasoning-Prompts/README.md), revisited here as the test-time-scaling mechanism itself

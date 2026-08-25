# Reward Modeling

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `02-Reward-Modeling`

## Why this matters

[Lesson 1](../01-The-Alignment-Problem/README.md) established that a model needs feedback beyond "here is one example of a good response" (which is all [SFT](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md) provides) — it needs to know *how much better* one response is than another, across many possible responses it might generate on its own. A **reward model (RM)** is the component that turns human judgment into a differentiable scalar signal a training algorithm can actually optimize. Every later lesson in this phase depends on it or on an alternative to it: [Lesson 3 (RLHF with PPO)](../03-RLHF-with-PPO/README.md) uses the RM's score directly as the reward signal for reinforcement learning; [Lesson 4 (DPO)](../04-Direct-Preference-Optimization-DPO/README.md) shows how to skip training an explicit RM at all while still optimizing the same underlying objective derived here; [Lesson 5 (RLAIF)](../05-RLAIF-and-Constitutional-AI/README.md) replaces the human labeler in this lesson's pipeline with an AI labeler but keeps everything else about reward modeling unchanged.

## What this lesson covers

- Why preference data is collected as pairwise comparisons rather than absolute scores
- The Bradley-Terry model of preferences, and where its loss function comes from
- Reward model architecture: turning a language model into a scalar scorer
- The reward-model training loop
- Known limitations: reward hacking and the reward model's own capacity ceiling

## 1. Why pairwise comparisons, not absolute scores

The naive approach to collecting human feedback would be to ask a labeler to rate a single response on an absolute scale (say, 1 to 10) for "helpfulness." In practice this is one of the least reliable ways to collect preference data: different labelers use the scale differently, the same labeler is inconsistent with themselves across sessions, and small wording differences in the prompt can shift what "a 7" means. **Pairwise comparison** — showing a labeler two (or more) candidate responses to the *same* prompt and asking only "which one is better?" — is dramatically more consistent and reliable to collect, because it is a much easier and better-calibrated cognitive task: humans are far better at relative judgments ("A is better than B") than absolute ones ("A is a 7.3"). This is the data-collection choice InstructGPT (Ouyang et al., 2022) and essentially every subsequent RLHF pipeline made, and it is the reason the entire mathematical framework below is built around comparisons instead of scores.

## 2. The Bradley-Terry model of preferences

Given a prompt `x` and two candidate responses `y_w` ("winner," i.e. the one the human preferred) and `y_l` ("loser"), we want to fit a scalar reward function `r(x, y)` such that responses humans prefer get higher scores. The **Bradley-Terry model** (a classical statistical model of paired comparisons, originally developed for ranking chess/sports competitors) assumes the probability that `y_w` is preferred over `y_l` is a logistic function of the *difference* in their underlying scores:

```
P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )
                 = exp(r(x, y_w)) / ( exp(r(x, y_w)) + exp(r(x, y_l)) )
```

This has an intuitive shape: if `r(x, y_w)` is much larger than `r(x, y_l)`, the model predicts near-certainty that a human would pick `y_w`; if the two scores are close, it predicts something close to a coin flip. Crucially, **only the difference between scores matters** — the Bradley-Terry model is invariant to adding any constant to both `r(x, y_w)` and `r(x, y_l)`, which is why a trained reward model's raw scores are only meaningful in comparison to each other, never as an absolute, calibrated quantity.

## 3. The reward model loss

We train a neural network `r_theta(x, y)` (parameterized by weights `theta`) to maximize the likelihood the Bradley-Terry model assigns to the human's actual observed choice on every collected pair `(x, y_w, y_l)`. Taking the negative log-likelihood over a dataset of `N` comparisons gives the reward-model loss:

```
loss(theta) = - E_{(x, y_w, y_l)} [ log( sigmoid( r_theta(x, y_w) - r_theta(x, y_l) ) ) ]
```

This is exactly binary cross-entropy, where the "label" is always 1 (the winner is always presented as the winner) and the "logit" is the score difference `r_theta(x, y_w) - r_theta(x, y_l)`. Minimizing it pushes `r_theta(x, y_w)` up and `r_theta(x, y_l)` down whenever the model gets a pair wrong or is under-confident, and does so *relatively* — the RM never needs, and is never given, an absolute "correct" score for any single response, only which of two is better. `example.py` implements this loss from scratch and verifies it actually recovers a sensible ranking.

## 4. Reward model architecture

In practice, a reward model is built by taking a pretrained (and usually SFT'd) language model and **replacing its language-modeling head with a single scalar output head** — instead of projecting the final hidden state to a distribution over the vocabulary (as in every model in this course so far), it projects to one number:

```
hidden_states = Transformer(x, y)          # same backbone as the policy model
pooled = hidden_states[:, -1, :]           # e.g. the final token's hidden state
reward = Linear(d_model -> 1)(pooled)      # a single scalar score for this (x, y) pair
```

Starting from a pretrained/SFT backbone rather than a randomly initialized network is important: the RM needs to genuinely understand language, factuality, and task quality to judge responses well, and re-using an already-trained model's representations is far more sample-efficient than learning language understanding from scratch on a comparatively small preference dataset. `example.py` uses a small MLP as a stand-in "scorer head" over toy fixed-length feature vectors, to keep the demo fast and focused purely on the Bradley-Terry loss mechanics rather than on training a full Transformer backbone.

## 5. Known limitations

- **Reward hacking**: since the RM is only an approximation of true human preference, a downstream optimizer (like PPO in [Lesson 3](../03-RLHF-with-PPO/README.md)) can find responses that score highly on the RM without actually being good — exploiting quirks or blind spots in what the RM learned rather than genuinely satisfying the human intent it was meant to approximate.
- **Distribution shift**: the RM is trained on comparisons of responses from *some* policy (often the SFT model). As RL fine-tuning pushes the policy's outputs into new regions of response-space, the RM's judgments there become progressively less reliable, since it never saw comparisons of that kind of output during its own training.
- **Label noise and disagreement**: real human labelers disagree with each other a meaningful fraction of the time on genuinely ambiguous comparisons, which puts a ceiling on how confidently *any* reward model can be trained, no matter how large.

## Video Script Outline

1. Motivation — turning "humans have preferences" into "a number a training loop can optimize"
2. Why pairwise comparisons beat absolute ratings for data collection
3. The Bradley-Terry model: preference probability as a sigmoid of a score difference
4. Deriving the reward-model loss as negative log-likelihood under Bradley-Terry
5. Architecture: language model backbone + scalar head, replacing the LM head
6. Walkthrough of `example.py` — synthetic preference pairs from a known ground truth, train the RM, verify its learned ranking correlates with the ground truth
7. Limitations: reward hacking and distribution shift, setting up Lesson 3's KL penalty

## Further Reading

- Bradley, Terry (1952), *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons* (the original statistical model)
- Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences* (the modern deep-RL reward-modeling-from-preferences paper)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — reward model training as stage 2 of the full RLHF pipeline)
- Stiennon et al. (2020), *Learning to Summarize from Human Feedback*

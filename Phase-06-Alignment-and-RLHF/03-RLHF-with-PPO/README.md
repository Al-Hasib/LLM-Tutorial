# RLHF with PPO

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `03-RLHF-with-PPO`

## Why this matters

This lesson assembles the full three-stage RLHF pipeline that turns a pretrained language model into something like a modern chat assistant, using every piece built so far in this course: [SFT](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md) for stage 1, the [reward model](../02-Reward-Modeling/README.md) from Lesson 2 for stage 2, and reinforcement learning for stage 3, which is the focus here. This is also the historical anchor of the whole phase — RLHF-with-PPO is the recipe InstructGPT and the original ChatGPT used, and understanding *why* it needs a KL penalty and a clipped objective (rather than plain policy gradient) sets up exactly why [Lesson 4&#39;s DPO](../04-Direct-Preference-Optimization-DPO/README.md) was such an appealing simplification when it arrived.

## What this lesson covers

- The full 3-stage RLHF pipeline, end to end
- Treating text generation as a reinforcement learning problem
- Why a raw policy-gradient update on the reward model's score alone leads to reward hacking
- The KL penalty against a frozen reference model, and why it fixes this
- PPO's clipped surrogate objective, at a conceptual level
- A hands-on demonstration of reward improving under a bounded KL budget, and degenerating without one

## 1. The full 3-stage pipeline

1. **Supervised fine-tuning (SFT)** — [Phase 05 Lesson 4](../../Phase-05-Finetuning-LLMs/04-Instruction-Tuning-SFT/README.md): fine-tune the pretrained base model on curated instruction/response pairs. This produces the starting policy for RL, and it also becomes the frozen **reference model** used below.
2. **Reward modeling** — [Lesson 2](../02-Reward-Modeling/README.md): collect pairwise human preferences over the SFT model's outputs, train a reward model `r_phi(x, y)` with the Bradley-Terry loss to predict which of two responses a human would prefer.
3. **RL fine-tuning (this lesson)** — further train the SFT model (now called the "policy," `pi_theta`) so that it generates responses that score highly under `r_phi`, using reinforcement learning.

## 2. Framing text generation as reinforcement learning

To apply RL, we map generation onto the standard RL vocabulary:

```
state       = the prompt plus whatever tokens have been generated so far
action      = the next token to generate
policy      = pi_theta(action | state)  -- the language model itself
episode     = one full generated response, start to end-of-sequence
reward      = r_phi(prompt, full response)  -- the reward model's score, given ONLY once, at the end
```

The reward is **sparse** (given only once per whole generated sequence, not per token) and comes entirely from the learned reward model. Reinforcement learning's job is to adjust `pi_theta`'s parameters so that it becomes more likely to generate the token sequences that lead to high reward.

## 3. Why plain policy gradient breaks: reward hacking

The most direct RL approach is a policy gradient update: increase the probability of token sequences that scored well, decrease the probability of ones that scored poorly, weighted by the reward. Applied naively against the reward model *alone*, this reliably breaks in practice. The reward model is only an approximation of true human preference (Lesson 2 §5), trained on a fixed dataset of past responses — it was never asked to judge every conceivable string of tokens, and it has systematic blind spots and quirks precisely because it is itself a trained neural network with finite capacity. A policy optimized purely to maximize `r_phi`'s output will, given enough freedom, drift toward exploiting those blind spots: producing text that scores unusually well on the reward model's quirks (repeated reassuring phrases, unusual token sequences, degenerate loops) rather than text that is *actually* better by the human standard the RM was meant to approximate. This failure mode is called **reward hacking** (or reward over-optimization), and it gets worse — not better — the harder you optimize against a fixed, imperfect reward model.

## 4. The fix: a KL penalty against the frozen reference model

The standard fix is to subtract a penalty from the reward for how far the current policy has drifted from the original SFT model (the frozen "reference" policy, `pi_ref`), measured with **KL divergence** at every generated token:

```
reward_total(x, y) = r_phi(x, y) - beta * KL( pi_theta(. | x, y_<t) || pi_ref(. | x, y_<t) )
```

Because `pi_ref` is frozen and known to produce fluent, sensible, human-like language (that's exactly what SFT trained it to do), this penalty acts as an anchor: the policy is free to shift its behavior toward whatever the reward model rewards, but only as far as the KL budget (controlled by `beta`) allows before the penalty outweighs the reward-model gain. This is precisely the mechanism that keeps RLHF-tuned models producing recognizable, grammatical language rather than degenerating into reward-model-exploiting gibberish. `example.py` demonstrates this directly: training with the KL term keeps divergence from the reference policy bounded while reward still improves; **removing the KL term entirely reproduces reward hacking on purpose**, as an honest illustration of what the penalty is protecting against.

## 5. PPO's clipped surrogate objective

Proximal Policy Optimization (Schulman et al., 2017) is the specific RL algorithm almost universally used for the RL stage, because plain policy gradient updates can be destructively large: a single bad update can shift the policy so far that all the following updates are computed under a policy distribution wildly different from the one the data was actually collected under, destabilizing training. PPO addresses this with a **clipped surrogate objective**: it computes the probability ratio between the new and old policy for the action actually taken,

```
ratio(theta) = pi_theta(a | s) / pi_theta_old(a | s)
```

and clips it to a small trust region `[1 - epsilon, 1 + epsilon]` (typically `epsilon = 0.2`) before multiplying by the estimated advantage `A`:

```
L_CLIP(theta) = E[ min( ratio(theta) * A,  clip(ratio(theta), 1-epsilon, 1+epsilon) * A ) ]
```

Taking the `min` of the unclipped and clipped versions means that whenever an update *would* push the ratio outside the trust region in a way that increases the objective, the clipped term caps the incentive — the update simply cannot claim credit for moving further than the trust region allows. This keeps every individual policy update conservative and close to the data it was actually estimated from, which in practice makes PPO dramatically more stable than plain policy gradient methods, at the cost of some implementation complexity (the "old" policy snapshot, the advantage estimator, several epochs of mini-batch updates per batch of rollouts).

## 6. Putting it together

Each RLHF-with-PPO training iteration, at a high level:

```
1. Sample prompts, generate responses with the current policy pi_theta      (rollout)
2. Score each response with the reward model r_phi                         (reward)
3. Compute the per-token KL penalty against the frozen reference pi_ref    (KL penalty)
4. Combine into a total reward, estimate advantages
5. Update pi_theta using PPO's clipped surrogate objective                 (policy update)
6. Repeat
```

`example.py` implements a simplified version of exactly this loop on a small toy sequential-generation task, with a real clipped surrogate objective and a real KL penalty against a frozen copy of the initial policy.

## Video Script Outline

1. Motivation — "we have an SFT model and a reward model; how do we actually use the second to improve the first?"
2. Framing text generation as an RL problem: states, actions, sparse episode-level reward
3. Why optimizing the reward model directly leads to reward hacking
4. The KL penalty against a frozen reference policy, and why it works
5. PPO's clipped surrogate objective, explained conceptually with the trust-region intuition
6. Walkthrough of `example.py` — reward improves, KL stays bounded, with the KL term
7. The same run with the KL weight set to zero — reward hacking, reproduced on purpose
8. Recap + preview of Lesson 4, which removes the RL loop (and the reward model!) entirely

## Further Reading

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT — the full pipeline this lesson implements a simplified version of)
- Stiennon et al. (2020), *Learning to Summarize from Human Feedback* (an earlier, detailed RLHF-with-PPO case study, including explicit discussion of the KL penalty and reward over-optimization)
- Schulman et al. (2015), *Trust Region Policy Optimization* (TRPO — PPO's predecessor and the origin of the trust-region idea)

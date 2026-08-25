# Direct Preference Optimization (DPO)

**Phase:** [Alignment and RLHF](../README.md) · **Topic folder:** `04-Direct-Preference-Optimization-DPO`

## Why this matters

[Lesson 2](../02-Reward-Modeling/README.md) derived the Bradley-Terry objective for turning pairwise human preferences into a scalar reward model, and [Lesson 3](../03-RLHF-with-PPO/README.md) spent an entire pipeline stage on the machinery needed to optimize a policy against that reward model with reinforcement learning: rollouts, a KL penalty against a frozen reference model, and PPO's clipped surrogate objective. **Direct Preference Optimization** (Rafailov et al., 2023) asks a sharp question: if all we ultimately want is a policy that satisfies the *same* Bradley-Terry preference objective, do we actually need the reward model and the RL loop as separate pieces of machinery, or can the algebra be rearranged so a single supervised-style loss gets us there directly? It turns out the algebra rearranges cleanly — DPO reaches the identical optimum as RLHF-with-PPO (under the same KL-constrained objective) while removing an entire training stage. This lesson is a direct payoff of Lessons 2 and 3: understanding DPO requires understanding exactly what it is that it is skipping. It also sets up [Lesson 5 (RLAIF and Constitutional AI)](../05-RLAIF-and-Constitutional-AI/README.md), where either RLHF-with-PPO or DPO can serve as the underlying optimizer once the preference *labels* come from an AI rather than a human.

## What this lesson covers

- The KL-constrained RLHF objective, restated, and the closed-form optimal policy it implies
- How that closed-form solution lets you substitute for the reward model algebraically, in terms of the policy itself
- The DPO loss, term by term, and why it is exactly Lesson 2's Bradley-Terry loss with an *implicit* reward
- What DPO removes from the Lesson 3 pipeline, and what it keeps
- A hands-on demonstration: training a tiny language model with only the DPO loss, no reward model, no RL rollouts

## 1. The RLHF objective and its closed-form optimal policy

Recall the KL-constrained objective RLHF (Lesson 3) optimizes: maximize the reward model's score while staying close to a reference policy `pi_ref`,

```
maximize_theta   E_{y ~ pi_theta(.|x)} [ r(x, y) ]  -  beta * KL( pi_theta(.|x) || pi_ref(.|x) )
```

This objective has a known closed-form solution (a standard result from KL-regularized control / maximum-entropy RL): the optimal policy is the reference policy re-weighted by the exponentiated reward,

```
pi*(y|x) = ( 1 / Z(x) ) * pi_ref(y|x) * exp( r(x, y) / beta )
```

where `Z(x) = sum_y pi_ref(y|x) * exp(r(x,y)/beta)` is an (intractable, per-prompt) normalizing constant. PPO exists precisely because this equation cannot be used directly — nobody can enumerate every possible response `y` to compute `Z(x)`, so Lesson 3 instead optimizes the objective indirectly with sampled rollouts and gradient ascent.

## 2. The key trick: solving for the reward instead of the policy

DPO's insight is to invert this equation algebraically. Rearranging for `r(x, y)`:

```
r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) )  +  beta * log Z(x)
```

Every reward function `r(x, y)` can therefore be re-expressed in terms of *some* policy `pi*` and the reference policy — the log-ratio between them, scaled by `beta`, plus a term that depends only on the prompt `x` (not on `y`). Now substitute this expression for `r(x, y_w)` and `r(x, y_l)` into the Bradley-Terry preference model from [Lesson 2 section 2](../02-Reward-Modeling/README.md#2-the-bradley-terry-model-of-preferences):

```
P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )
```

The `beta * log Z(x)` terms are identical for `y_w` and `y_l` (both are responses to the *same* prompt `x`) and **cancel exactly** in the subtraction. What's left is a preference probability expressed purely in terms of a policy's log-probabilities and the reference model's log-probabilities — no reward model, and no intractable `Z(x)`, anywhere in sight.

## 3. The DPO loss

Substituting that cancellation into the Bradley-Terry negative log-likelihood (exactly [Lesson 2 section 3](../02-Reward-Modeling/README.md#3-the-reward-model-loss)'s loss, but now written directly in terms of a *policy* `pi_theta` standing in for the optimal `pi*`) gives the DPO loss:

```
loss(theta) = - E_{(x, y_w, y_l)} [ log( sigmoid(
                  beta * ( log( pi_theta(y_w|x) / pi_ref(y_w|x) )
                          - log( pi_theta(y_l|x) / pi_ref(y_l|x) ) )
              ) ) ]
```

Term by term:

- `pi_theta` — the policy currently being trained (initialized from the SFT model, exactly like the RL policy in Lesson 3).
- `pi_ref` — a **frozen** copy of the SFT model, used only for scoring, never updated. Its role is identical to the reference model in Lesson 3's KL penalty.
- `beta` — a temperature controlling how sharply the loss responds to a given log-ratio margin; it plays exactly the role Lesson 3's KL-penalty weight `beta` played in the original objective (section 1), because it literally is that same `beta`, algebraically threaded through.
- `log( pi_theta(y|x) / pi_ref(y|x) )` — the **implicit reward**: DPO never trains a network whose whole job is to output a scalar score, but this log-ratio *is* mathematically a reward function per section 2, and it moves up for `y_w` and down for `y_l` exactly the way an explicit reward model's score would.

This loss is differentiable end to end in `pi_theta`'s parameters using nothing but a forward pass through the policy (and the frozen reference) on each preference triple — ordinary supervised-style gradient descent, no sampling, no rollouts, no advantage estimation, no PPO clipping.

## 4. Why this works: an implicit reward model, hiding inside the policy

The intuitive way to see DPO is: **the ratio between the current policy and the reference policy, at any given response, already behaves exactly like a reward score for that response** — responses the policy has learned to make more likely than the reference model would have made them are, by this equation, responses of higher implicit reward, and vice versa. Training directly on the Bradley-Terry loss with this implicit reward pushes `pi_theta` to raise `pi_theta(y_w|x)/pi_ref(y_w|x)` relative to `pi_theta(y_l|x)/pi_ref(y_l|x)` for every observed preference pair. Because this is the exact same Bradley-Terry objective Lesson 2's explicit reward model was trained to satisfy, and because section 1's closed-form result shows the two objectives share the same optimum, **DPO converges to the same policy RLHF-with-PPO was trying to reach — it simply reparameterizes the problem so the reward model and the RL optimizer both disappear**, replaced by one loss computed directly on log-probabilities the policy and reference model already know how to produce. `example.py` verifies this concretely: it trains a tiny model with only this loss and confirms the resulting log-probability margin between chosen and rejected responses grows exactly as the theory predicts.

## 5. What DPO removes, and what it keeps

| | RLHF with PPO (Lesson 3) | DPO |
|---|---|---|
| Separate reward model | Yes, trained first (Lesson 2) | No — implicit, defined by the policy/reference log-ratio |
| Sampling / rollouts during training | Yes, every iteration | No — trains on a fixed, static dataset of triples |
| RL algorithm (PPO, advantage estimation, clipping) | Yes | No — a single supervised-style loss |
| Reference model | Yes (KL penalty) | Yes (same role, algebraically folded into the loss) |
| `beta` hyperparameter | KL penalty weight | Same role, same name, same effect |
| Training stability concerns | Rollout variance, reward hacking, PPO tuning | Fewer moving parts, but sensitive to `beta` and to how far `pi_theta` drifts from the distribution the preference data was collected under |

DPO does **not** remove the need for preference data, and it does not remove the frozen reference model — it removes the reward-model-training stage and the entire RL loop built around it, which in practice is most of the implementation complexity and compute cost of Lesson 3.

## Video Script Outline

1. Motivation — "we have a Bradley-Terry objective and a KL-constrained RL objective; must they be two separate training stages?"
2. The KL-constrained RLHF objective and its closed-form optimal policy (section 1)
3. The algebraic trick: solving for the reward in terms of the policy, and substituting into Bradley-Terry (section 2)
4. The DPO loss, explained term by term: pi_theta, pi_ref, beta, and the implicit reward (section 3)
5. Why this reaches the same optimum as RLHF-with-PPO, intuitively (section 4)
6. Side-by-side comparison table: what DPO removes vs keeps relative to Lesson 3 (section 5)
7. Walkthrough of `example.py` — train a tiny GPT purely with the DPO loss on synthetic preference triples, watch the chosen-vs-rejected log-probability margin grow with no reward model and no RL rollout anywhere in the loop
8. Recap + preview of Lesson 5, where the preference *labels themselves* stop coming from humans

## Further Reading

- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* (the algorithm DPO's derivation shows how to avoid)
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (the RLHF pipeline DPO simplifies)
- Azar et al. (2023), *A General Theoretical Paradigm to Understand Learning from Human Preferences* (a follow-up analysis of DPO's objective and its assumptions)
